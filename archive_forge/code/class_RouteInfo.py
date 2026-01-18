from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouteInfo(_messages.Message):
    """For display only. Metadata associated with a Compute Engine route.

  Enums:
    NextHopTypeValueValuesEnum: Type of next hop.
    RouteScopeValueValuesEnum: Indicates where route is applicable.
    RouteTypeValueValuesEnum: Type of route.

  Fields:
    destIpRange: Destination IP range of the route.
    destPortRanges: Destination port ranges of the route. Policy based routes
      only.
    displayName: Name of a route.
    instanceTags: Instance tags of the route.
    nccHubUri: URI of a NCC Hub. NCC_HUB routes only.
    nccSpokeUri: URI of a NCC Spoke. NCC_HUB routes only.
    networkUri: URI of a Compute Engine network. NETWORK routes only.
    nextHop: Next hop of the route.
    nextHopType: Type of next hop.
    priority: Priority of the route.
    protocols: Protocols of the route. Policy based routes only.
    routeScope: Indicates where route is applicable.
    routeType: Type of route.
    srcIpRange: Source IP address range of the route. Policy based routes
      only.
    srcPortRanges: Source port ranges of the route. Policy based routes only.
    uri: URI of a route. Dynamic, peering static and peering dynamic routes do
      not have an URI. Advertised route from Google Cloud VPC to on-premises
      network also does not have an URI.
  """

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

    class RouteScopeValueValuesEnum(_messages.Enum):
        """Indicates where route is applicable.

    Values:
      ROUTE_SCOPE_UNSPECIFIED: Unspecified scope. Default value.
      NETWORK: Route is applicable to packets in Network.
      NCC_HUB: Route is applicable to packets using NCC Hub's routing table.
    """
        ROUTE_SCOPE_UNSPECIFIED = 0
        NETWORK = 1
        NCC_HUB = 2

    class RouteTypeValueValuesEnum(_messages.Enum):
        """Type of route.

    Values:
      ROUTE_TYPE_UNSPECIFIED: Unspecified type. Default value.
      SUBNET: Route is a subnet route automatically created by the system.
      STATIC: Static route created by the user, including the default route to
        the internet.
      DYNAMIC: Dynamic route exchanged between BGP peers.
      PEERING_SUBNET: A subnet route received from peering network.
      PEERING_STATIC: A static route received from peering network.
      PEERING_DYNAMIC: A dynamic route received from peering network.
      POLICY_BASED: Policy based route.
    """
        ROUTE_TYPE_UNSPECIFIED = 0
        SUBNET = 1
        STATIC = 2
        DYNAMIC = 3
        PEERING_SUBNET = 4
        PEERING_STATIC = 5
        PEERING_DYNAMIC = 6
        POLICY_BASED = 7
    destIpRange = _messages.StringField(1)
    destPortRanges = _messages.StringField(2, repeated=True)
    displayName = _messages.StringField(3)
    instanceTags = _messages.StringField(4, repeated=True)
    nccHubUri = _messages.StringField(5)
    nccSpokeUri = _messages.StringField(6)
    networkUri = _messages.StringField(7)
    nextHop = _messages.StringField(8)
    nextHopType = _messages.EnumField('NextHopTypeValueValuesEnum', 9)
    priority = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    protocols = _messages.StringField(11, repeated=True)
    routeScope = _messages.EnumField('RouteScopeValueValuesEnum', 12)
    routeType = _messages.EnumField('RouteTypeValueValuesEnum', 13)
    srcIpRange = _messages.StringField(14)
    srcPortRanges = _messages.StringField(15, repeated=True)
    uri = _messages.StringField(16)