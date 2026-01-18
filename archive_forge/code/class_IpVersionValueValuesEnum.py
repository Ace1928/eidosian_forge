from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IpVersionValueValuesEnum(_messages.Enum):
    """Output only. The version of this IP address.

    Values:
      IP_VERSION_UNSPECIFIED: The version of this ip is unknown.
      IPV4: The ip is an IPv4 address.
      IPV6: The ip is an IPv6 address.
    """
    IP_VERSION_UNSPECIFIED = 0
    IPV4 = 1
    IPV6 = 2