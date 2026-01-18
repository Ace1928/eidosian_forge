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
class RemoveInstances(base.SilentCommand):
    """Removes resources from an unmanaged instance group by instance name.

    *{command}* removes instances from an unmanaged instance group using
  the instance name.

  This does not delete the actual instance resources but removes
  it from the instance group.
  """

    @staticmethod
    def Args(parser):
        RemoveInstances.ZONAL_INSTANCE_GROUP_ARG = instance_groups_flags.MakeZonalInstanceGroupArg()
        RemoveInstances.ZONAL_INSTANCE_GROUP_ARG.AddArgument(parser)
        parser.add_argument('--instances', required=True, type=arg_parsers.ArgList(min_length=1), metavar='INSTANCE', help='The names of the instances to remove from the instance group.')

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        group_ref = RemoveInstances.ZONAL_INSTANCE_GROUP_ARG.ResolveAsResource(args, holder.resources, default_scope=compute_scope.ScopeEnum.ZONE, scope_lister=flags.GetDefaultScopeLister(client))
        instance_references = []
        for instance in args.instances:
            ref = holder.resources.Parse(instance, params={'project': group_ref.project, 'zone': group_ref.zone}, collection='compute.instances')
            instance_references.append(ref)
        instance_groups_utils.ValidateInstanceInZone(instance_references, group_ref.zone)
        instance_references = [client.messages.InstanceReference(instance=inst.SelfLink()) for inst in instance_references]
        request_payload = client.messages.InstanceGroupsRemoveInstancesRequest(instances=instance_references)
        request = client.messages.ComputeInstanceGroupsRemoveInstancesRequest(instanceGroup=group_ref.Name(), instanceGroupsRemoveInstancesRequest=request_payload, zone=group_ref.zone, project=group_ref.project)
        return client.MakeRequests([(client.apitools_client.instanceGroups, 'RemoveInstances', request)])