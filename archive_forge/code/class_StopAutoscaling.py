from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
class StopAutoscaling(base.Command):
    """Stop autoscaling a managed instance group.

    *{command}* stops autoscaling a managed instance group and deletes the
  autoscaler configuration. If autoscaling is not enabled for the managed
  instance group, this command does nothing and will report an error.

  If you need to keep the autoscaler configuration, you can temporarily disable
  an autoscaler by setting its `mode` to `off` using the ``update-autoscaling''
  command instead.

  """

    @staticmethod
    def Args(parser):
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)

    def CreateGroupReference(self, client, resources, args):
        resource_arg = instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG
        default_scope = compute_scope.ScopeEnum.ZONE
        scope_lister = flags.GetDefaultScopeLister(client)
        return resource_arg.ResolveAsResource(args, resources, default_scope=default_scope, scope_lister=scope_lister)

    def GetAutoscalerServiceForGroup(self, client, group_ref):
        if _IsZonalGroup(group_ref):
            return client.apitools_client.autoscalers
        else:
            return client.apitools_client.regionAutoscalers

    def ScopeRequest(self, request, igm_ref):
        if _IsZonalGroup(igm_ref):
            request.zone = igm_ref.zone
        else:
            request.region = igm_ref.region

    def GetAutoscalerResource(self, client, resources, igm_ref, args):
        if _IsZonalGroup(igm_ref):
            scope_type = 'zone'
            location = managed_instance_groups_utils.CreateZoneRef(resources, igm_ref)
            zones, regions = ([location], None)
        else:
            scope_type = 'region'
            location = managed_instance_groups_utils.CreateRegionRef(resources, igm_ref)
            zones, regions = (None, [location])
        autoscaler = managed_instance_groups_utils.AutoscalerForMig(mig_name=args.name, autoscalers=managed_instance_groups_utils.AutoscalersForLocations(regions=regions, zones=zones, client=client), location=location, scope_type=scope_type)
        if autoscaler is None:
            raise managed_instance_groups_utils.ResourceNotFoundException('The managed instance group is not autoscaled.')
        return autoscaler

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        igm_ref = self.CreateGroupReference(client, holder.resources, args)
        service = self.GetAutoscalerServiceForGroup(client, igm_ref)
        managed_instance_groups_utils.GetInstanceGroupManagerOrThrow(igm_ref, client)
        autoscaler = self.GetAutoscalerResource(client, holder.resources, igm_ref, args)
        request = service.GetRequestType('Delete')(project=igm_ref.project, autoscaler=autoscaler.name)
        self.ScopeRequest(request, igm_ref)
        return client.MakeRequests([(service, 'Delete', request)])