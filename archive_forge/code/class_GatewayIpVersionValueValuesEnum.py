from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GatewayIpVersionValueValuesEnum(_messages.Enum):
    """The IP family of the gateway IPs for the HA-VPN gateway interfaces. If
    not specified, IPV4 will be used.

    Values:
      IPV4: Every HA-VPN gateway interface is configured with an IPv4 address.
      IPV6: Every HA-VPN gateway interface is configured with an IPv6 address.
    """
    IPV4 = 0
    IPV6 = 1