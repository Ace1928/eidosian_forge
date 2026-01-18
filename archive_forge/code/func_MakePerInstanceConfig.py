from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def MakePerInstanceConfig(messages, name, preserved_state_disks, preserved_state_metadata, preserved_internal_ips=None, preserved_external_ips=None):
    """Make a per-instance config message from preserved state.

  Args:
    messages: Compute API messages
    name: Name of the instance
    preserved_state_disks: List of preserved state disk map entries
    preserved_state_metadata: List of preserved state metadata map entries
    preserved_internal_ips: List of preserved internal IPs
    preserved_external_ips: List of preserved external IPs

  Returns:
    Per-instance config message
  """
    return messages.PerInstanceConfig(name=name, preservedState=MakePreservedState(messages, preserved_state_disks, preserved_state_metadata, preserved_internal_ips, preserved_external_ips))