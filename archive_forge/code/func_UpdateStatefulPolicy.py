from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.instance_groups import flags
def UpdateStatefulPolicy(messages, stateful_policy_to_update, preserved_state_disks=None, preserved_state_internal_ips=None, preserved_state_external_ips=None):
    """Update stateful policy proto from a list of preserved state attributes."""
    if preserved_state_disks is not None:
        stateful_policy_to_update.preservedState.disks = messages.StatefulPolicyPreservedState.DisksValue(additionalProperties=preserved_state_disks)
    if preserved_state_internal_ips is not None:
        stateful_policy_to_update.preservedState.internalIPs = messages.StatefulPolicyPreservedState.InternalIPsValue(additionalProperties=preserved_state_internal_ips)
    if preserved_state_external_ips is not None:
        stateful_policy_to_update.preservedState.externalIPs = messages.StatefulPolicyPreservedState.ExternalIPsValue(additionalProperties=preserved_state_external_ips)
    return stateful_policy_to_update