from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceCidrValueValuesEnum(_messages.Enum):
    """Service CIDR, if any.

    Values:
      SERVICE_CIDR_UNSPECIFIED: Unspecified value.
      DISABLED: Services are disabled for the given network.
      HIGH_26: Use the highest /26 block of the network to host services.
      HIGH_27: Use the highest /27 block of the network to host services.
      HIGH_28: Use the highest /28 block of the network to host services.
    """
    SERVICE_CIDR_UNSPECIFIED = 0
    DISABLED = 1
    HIGH_26 = 2
    HIGH_27 = 3
    HIGH_28 = 4