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
def _GetPatchForStatefulPolicyForInternalIPs(self, client, update_internal_ips=None, remove_interface_names=None):
    return self._GetStatefulPolicyPatchForStatefulIPsCommon(client, functools.partial(policy_utils.MakeInternalIPEntry, client.messages), functools.partial(policy_utils.MakeInternalIPNullEntryForDisablingInPatch, client), update_internal_ips, remove_interface_names)