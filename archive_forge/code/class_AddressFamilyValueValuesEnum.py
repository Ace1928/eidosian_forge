from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddressFamilyValueValuesEnum(_messages.Enum):
    """(Required) limit results to this address family (either IPv4 or IPv6)

    Values:
      IPV4: <no description>
      IPV6: <no description>
      UNSPECIFIED_IP_VERSION: <no description>
    """
    IPV4 = 0
    IPV6 = 1
    UNSPECIFIED_IP_VERSION = 2