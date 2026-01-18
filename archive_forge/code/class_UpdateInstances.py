from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_groups_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as instance_groups_managed_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import update_instances_utils
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class UpdateInstances(base.Command):
    """Immediately update selected instances in a Compute Engine managed instance group."""

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat('\n        table(project(),\n              zone(),\n              instanceName:label=INSTANCE,\n              status)')
        instance_groups_managed_flags.AddUpdateInstancesArgs(parser=parser)
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        igm_ref = instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.ResolveAsResource(args, holder.resources, default_scope=compute_scope.ScopeEnum.ZONE, scope_lister=flags.GetDefaultScopeLister(client))
        update_instances_utils.ValidateIgmReference(igm_ref)
        if igm_ref.Collection() == 'compute.instanceGroupManagers':
            minimal_action = update_instances_utils.ParseInstanceActionFlag('--minimal-action', args.minimal_action or 'none', client.messages.InstanceGroupManagersApplyUpdatesRequest.MinimalActionValueValuesEnum)
            most_disruptive_allowed_action = update_instances_utils.ParseInstanceActionFlag('--most-disruptive-allowed-action', args.most_disruptive_allowed_action or 'replace', client.messages.InstanceGroupManagersApplyUpdatesRequest.MostDisruptiveAllowedActionValueValuesEnum)
            instances_holder_field = 'instanceGroupManagersApplyUpdatesRequest'
            request = self._CreateZonalApplyUpdatesRequest(igm_ref, minimal_action, most_disruptive_allowed_action, client)
        elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
            minimal_action = update_instances_utils.ParseInstanceActionFlag('--minimal-action', args.minimal_action or 'none', client.messages.RegionInstanceGroupManagersApplyUpdatesRequest.MinimalActionValueValuesEnum)
            most_disruptive_allowed_action = update_instances_utils.ParseInstanceActionFlag('--most-disruptive-allowed-action', args.most_disruptive_allowed_action or 'replace', client.messages.RegionInstanceGroupManagersApplyUpdatesRequest.MostDisruptiveAllowedActionValueValuesEnum)
            instances_holder_field = 'regionInstanceGroupManagersApplyUpdatesRequest'
            request = self._CreateRegionalApplyUpdatesRequest(igm_ref, minimal_action, most_disruptive_allowed_action, client)
        else:
            raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))
        if args.all_instances:
            return instance_groups_utils.SendAllInstancesRequest(api_holder=holder, method_name='ApplyUpdatesToInstances', request_template=request, all_instances_holder_field=instances_holder_field, igm_ref=igm_ref)
        else:
            return instance_groups_utils.SendInstancesRequestsAndPostProcessOutputs(api_holder=holder, method_name='ApplyUpdatesToInstances', request_template=request, instances_holder_field=instances_holder_field, igm_ref=igm_ref, instances=args.instances)

    def _CreateZonalApplyUpdatesRequest(self, igm_ref, minimal_action, most_disruptive_allowed_action, client):
        return client.messages.ComputeInstanceGroupManagersApplyUpdatesToInstancesRequest(instanceGroupManager=igm_ref.Name(), instanceGroupManagersApplyUpdatesRequest=client.messages.InstanceGroupManagersApplyUpdatesRequest(instances=[], minimalAction=minimal_action, mostDisruptiveAllowedAction=most_disruptive_allowed_action), project=igm_ref.project, zone=igm_ref.zone)

    def _CreateRegionalApplyUpdatesRequest(self, igm_ref, minimal_action, most_disruptive_allowed_action, client):
        return client.messages.ComputeRegionInstanceGroupManagersApplyUpdatesToInstancesRequest(instanceGroupManager=igm_ref.Name(), regionInstanceGroupManagersApplyUpdatesRequest=client.messages.RegionInstanceGroupManagersApplyUpdatesRequest(instances=[], minimalAction=minimal_action, mostDisruptiveAllowedAction=most_disruptive_allowed_action), project=igm_ref.project, region=igm_ref.region)