from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.instance_groups import flags
def MakeStatefulPolicy(messages, preserved_state_disks):
    """Make stateful policy proto from a list of preserved state disk protos."""
    if not preserved_state_disks:
        preserved_state_disks = []
    return messages.StatefulPolicy(preservedState=messages.StatefulPolicyPreservedState(disks=messages.StatefulPolicyPreservedState.DisksValue(additionalProperties=preserved_state_disks)))