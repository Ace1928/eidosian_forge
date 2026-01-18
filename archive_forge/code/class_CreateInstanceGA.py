from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_configs_messages
@base.ReleaseTracks(base.ReleaseTrack.GA)
class CreateInstanceGA(base.CreateCommand):
    """Create a new virtual machine instance in a managed instance group."""

    @classmethod
    def Args(cls, parser):
        instance_groups_flags.GetInstanceGroupManagerArg(region_flag=True).AddArgument(parser, operation_type='create instance in')
        instance_groups_flags.AddCreateInstancesFlags(parser)

    @staticmethod
    def _CreateNewInstanceReference(holder, igm_ref, instance_name):
        """Creates reference to instance in instance group (zonal or regional)."""
        if igm_ref.Collection() == 'compute.instanceGroupManagers':
            instance_ref = holder.resources.Parse(instance_name, params={'project': igm_ref.project, 'zone': igm_ref.zone}, collection='compute.instances')
        elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
            instance_ref = holder.resources.Parse(instance_name, params={'project': igm_ref.project, 'zone': igm_ref.region + '-a'}, collection='compute.instances')
        else:
            raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))
        if not instance_ref:
            raise managed_instance_groups_utils.ResourceCannotBeResolvedException('Instance name {0} cannot be resolved.'.format(instance_name))
        return instance_ref

    def Run(self, args):
        self._ValidateStatefulFlagsForInstanceConfigs(args)
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        resources = holder.resources
        igm_ref = instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.ResolveAsResource(args, resources, scope_lister=compute_flags.GetDefaultScopeLister(client))
        instance_ref = self._CreateNewInstanceReference(holder=holder, igm_ref=igm_ref, instance_name=args.instance)
        per_instance_config_message = self._CreatePerInstanceConfgMessage(holder, instance_ref, args)
        operation_ref, service = instance_configs_messages.CallCreateInstances(holder=holder, igm_ref=igm_ref, per_instance_config_message=per_instance_config_message)
        operation_poller = poller.Poller(service)
        create_result = waiter.WaitFor(operation_poller, operation_ref, 'Creating instance.')
        return create_result

    def _ValidateStatefulFlagsForInstanceConfigs(self, args):
        instance_groups_flags.ValidateMigStatefulFlagsForInstanceConfigs(args, need_disk_source=True)
        instance_groups_flags.ValidateMigStatefulIPFlagsForInstanceConfigs(args, current_internal_addresses=[], current_external_addresses=[])

    def _CreatePerInstanceConfgMessage(self, holder, instance_ref, args):
        return instance_configs_messages.CreatePerInstanceConfigMessageWithIPs(holder, instance_ref, args.stateful_disk, args.stateful_metadata, args.stateful_internal_ip, args.stateful_external_ip, disk_getter=NonExistentDiskGetter())