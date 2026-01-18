from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_groups_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class SuspendInstances(base.Command):
    """Suspend instances owned by a managed instance group."""

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat('\n        table(project(),\n              zone(),\n              instanceName:label=INSTANCE,\n              status)')
        parser.add_argument('--instances', type=arg_parsers.ArgList(min_length=1), metavar='INSTANCE', required=True, help='Names of instances to suspend.')
        parser.add_argument('--force', default=False, action='store_true', help='\n          Immediately suspend the specified instances, skipping the initial\n          delay, if one is specified in the standby policy.')
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        resource_arg = instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG
        default_scope = compute_scope.ScopeEnum.ZONE
        scope_lister = flags.GetDefaultScopeLister(client)
        igm_ref = resource_arg.ResolveAsResource(args, holder.resources, default_scope=default_scope, scope_lister=scope_lister)
        if igm_ref.Collection() == 'compute.instanceGroupManagers':
            instances_holder_field = 'instanceGroupManagersSuspendInstancesRequest'
            request = client.messages.ComputeInstanceGroupManagersSuspendInstancesRequest(instanceGroupManager=igm_ref.Name(), instanceGroupManagersSuspendInstancesRequest=client.messages.InstanceGroupManagersSuspendInstancesRequest(instances=[]), project=igm_ref.project, zone=igm_ref.zone)
        elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
            instances_holder_field = 'regionInstanceGroupManagersSuspendInstancesRequest'
            request = client.messages.ComputeRegionInstanceGroupManagersSuspendInstancesRequest(instanceGroupManager=igm_ref.Name(), regionInstanceGroupManagersSuspendInstancesRequest=client.messages.RegionInstanceGroupManagersSuspendInstancesRequest(instances=[]), project=igm_ref.project, region=igm_ref.region)
        else:
            raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))
        if args.IsSpecified('force'):
            if igm_ref.Collection() == 'compute.instanceGroupManagers':
                request.instanceGroupManagersSuspendInstancesRequest.forceSuspend = args.force
            else:
                request.regionInstanceGroupManagersSuspendInstancesRequest.forceSuspend = args.force
        return instance_groups_utils.SendInstancesRequestsAndPostProcessOutputs(api_holder=holder, method_name='SuspendInstances', request_template=request, instances_holder_field=instances_holder_field, igm_ref=igm_ref, instances=args.instances)