from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import stateful_policy_utils as policy_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as managed_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def _GetUpdatedStatefulPolicy(self, client, current_stateful_policy, args):
    """Create an updated stateful policy based on specified args."""
    update_disks = args.stateful_disk
    remove_device_names = args.remove_stateful_disks
    stateful_disks = self._GetUpdatedStatefulPolicyForDisks(client, current_stateful_policy, update_disks, remove_device_names)
    stateful_policy = policy_utils.MakeStatefulPolicy(client.messages, stateful_disks)
    stateful_internal_ips = self._GetPatchForStatefulPolicyForInternalIPs(client, args.stateful_internal_ip, args.remove_stateful_internal_ips)
    stateful_external_ips = self._GetPatchForStatefulPolicyForExternalIPs(client, args.stateful_external_ip, args.remove_stateful_external_ips)
    return policy_utils.UpdateStatefulPolicy(client.messages, stateful_policy, None, stateful_internal_ips, stateful_external_ips)