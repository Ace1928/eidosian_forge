from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def MakePreservedState(messages, preserved_state_disks=None, preserved_state_metadata=None, preserved_internal_ips=None, preserved_external_ips=None):
    """Make preservedState message."""
    preserved_state = messages.PreservedState()
    if preserved_state_disks is not None:
        preserved_state.disks = messages.PreservedState.DisksValue(additionalProperties=preserved_state_disks)
    if preserved_state_metadata is not None:
        preserved_state.metadata = messages.PreservedState.MetadataValue(additionalProperties=preserved_state_metadata)
    if preserved_internal_ips is not None:
        preserved_state.internalIPs = messages.PreservedState.InternalIPsValue(additionalProperties=preserved_internal_ips)
    if preserved_external_ips is not None:
        preserved_state.externalIPs = messages.PreservedState.ExternalIPsValue(additionalProperties=preserved_external_ips)
    return preserved_state