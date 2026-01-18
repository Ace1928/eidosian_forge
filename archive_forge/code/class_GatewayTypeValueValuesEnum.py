from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GatewayTypeValueValuesEnum(_messages.Enum):
    """Indicates whether the device is a gateway.

    Values:
      GATEWAY_TYPE_UNSPECIFIED: If unspecified, the device is considered a
        non-gateway device.
      GATEWAY: The device is a gateway.
      NON_GATEWAY: The device is not a gateway.
    """
    GATEWAY_TYPE_UNSPECIFIED = 0
    GATEWAY = 1
    NON_GATEWAY = 2