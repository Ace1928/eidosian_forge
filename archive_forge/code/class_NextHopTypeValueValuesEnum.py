from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NextHopTypeValueValuesEnum(_messages.Enum):
    """Type of next hop.

    Values:
      NEXT_HOP_TYPE_UNSPECIFIED: Unspecified type. Default value.
      NEXT_HOP_IP: Next hop is an IP address.
      NEXT_HOP_INSTANCE: Next hop is a Compute Engine instance.
      NEXT_HOP_NETWORK: Next hop is a VPC network gateway.
      NEXT_HOP_PEERING: Next hop is a peering VPC.
      NEXT_HOP_INTERCONNECT: Next hop is an interconnect.
      NEXT_HOP_VPN_TUNNEL: Next hop is a VPN tunnel.
      NEXT_HOP_VPN_GATEWAY: Next hop is a VPN gateway. This scenario only
        happens when tracing connectivity from an on-premises network to
        Google Cloud through a VPN. The analysis simulates a packet departing
        from the on-premises network through a VPN tunnel and arriving at a
        Cloud VPN gateway.
      NEXT_HOP_INTERNET_GATEWAY: Next hop is an internet gateway.
      NEXT_HOP_BLACKHOLE: Next hop is blackhole; that is, the next hop either
        does not exist or is not running.
      NEXT_HOP_ILB: Next hop is the forwarding rule of an Internal Load
        Balancer.
      NEXT_HOP_ROUTER_APPLIANCE: Next hop is a [router appliance
        instance](https://cloud.google.com/network-connectivity/docs/network-
        connectivity-center/concepts/ra-overview).
      NEXT_HOP_NCC_HUB: Next hop is an NCC hub.
    """
    NEXT_HOP_TYPE_UNSPECIFIED = 0
    NEXT_HOP_IP = 1
    NEXT_HOP_INSTANCE = 2
    NEXT_HOP_NETWORK = 3
    NEXT_HOP_PEERING = 4
    NEXT_HOP_INTERCONNECT = 5
    NEXT_HOP_VPN_TUNNEL = 6
    NEXT_HOP_VPN_GATEWAY = 7
    NEXT_HOP_INTERNET_GATEWAY = 8
    NEXT_HOP_BLACKHOLE = 9
    NEXT_HOP_ILB = 10
    NEXT_HOP_ROUTER_APPLIANCE = 11
    NEXT_HOP_NCC_HUB = 12