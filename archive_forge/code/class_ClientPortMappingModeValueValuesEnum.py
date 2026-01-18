from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClientPortMappingModeValueValuesEnum(_messages.Enum):
    """Only valid when networkEndpointType is GCE_VM_IP_PORT and the NEG is
    regional.

    Values:
      CLIENT_PORT_PER_ENDPOINT: For each endpoint there is exactly one client
        port.
      PORT_MAPPING_DISABLED: NEG should not be used for mapping client port to
        destination.
    """
    CLIENT_PORT_PER_ENDPOINT = 0
    PORT_MAPPING_DISABLED = 1