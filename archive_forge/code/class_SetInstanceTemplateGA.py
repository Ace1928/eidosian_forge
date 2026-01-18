from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as managed_flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class SetInstanceTemplateGA(base.Command):
    """Command for setting instance template of managed instance group."""

    @classmethod
    def Args(cls, parser):
        managed_flags.INSTANCE_TEMPLATE_ARG.AddArgument(parser)
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        resource_arg = instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG
        default_scope = compute_scope.ScopeEnum.ZONE
        scope_lister = flags.GetDefaultScopeLister(client)
        igm_ref = resource_arg.ResolveAsResource(args, holder.resources, default_scope=default_scope, scope_lister=scope_lister)
        template_ref = managed_flags.INSTANCE_TEMPLATE_ARG.ResolveAsResource(args, holder.resources, default_scope=flags.compute_scope.ScopeEnum.GLOBAL)
        return self._MakePatchRequest(client, igm_ref, template_ref)

    def _MakePatchRequest(self, client, igm_ref, template_ref):
        messages = client.messages
        igm_resource = messages.InstanceGroupManager(instanceTemplate=template_ref.SelfLink(), versions=[messages.InstanceGroupManagerVersion(instanceTemplate=template_ref.SelfLink())])
        if igm_ref.Collection() == 'compute.instanceGroupManagers':
            service = client.apitools_client.instanceGroupManagers
            request_type = messages.ComputeInstanceGroupManagersPatchRequest
        elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
            service = client.apitools_client.regionInstanceGroupManagers
            request_type = messages.ComputeRegionInstanceGroupManagersPatchRequest
        else:
            raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))
        request = request_type(**igm_ref.AsDict())
        request.instanceGroupManagerResource = igm_resource
        return client.MakeRequests([(service, 'Patch', request)])