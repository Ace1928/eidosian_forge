from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterStatusBgpPeerStatus(_messages.Message):
    """A RouterStatusBgpPeerStatus object.

  Enums:
    StatusValueValuesEnum: Status of the BGP peer: {UP, DOWN}
    StatusReasonValueValuesEnum: Indicates why particular status was returned.

  Fields:
    advertisedRoutes: Routes that were advertised to the remote BGP peer
    bfdStatus: A BfdStatus attribute.
    enableIpv4: Enable IPv4 traffic over BGP Peer. It is enabled by default if
      the peerIpAddress is version 4.
    enableIpv6: Enable IPv6 traffic over BGP Peer. If not specified, it is
      disabled by default.
    ipAddress: IP address of the local BGP interface.
    ipv4NexthopAddress: IPv4 address of the local BGP interface.
    ipv6NexthopAddress: IPv6 address of the local BGP interface.
    linkedVpnTunnel: URL of the VPN tunnel that this BGP peer controls.
    md5AuthEnabled: Informs whether MD5 authentication is enabled on this BGP
      peer.
    name: Name of this BGP peer. Unique within the Routers resource.
    numLearnedRoutes: Number of routes learned from the remote BGP Peer.
    peerIpAddress: IP address of the remote BGP interface.
    peerIpv4NexthopAddress: IPv4 address of the remote BGP interface.
    peerIpv6NexthopAddress: IPv6 address of the remote BGP interface.
    routerApplianceInstance: [Output only] URI of the VM instance that is used
      as third-party router appliances such as Next Gen Firewalls, Virtual
      Routers, or Router Appliances. The VM instance is the peer side of the
      BGP session.
    state: The state of the BGP session. For a list of possible values for
      this field, see BGP session states.
    status: Status of the BGP peer: {UP, DOWN}
    statusReason: Indicates why particular status was returned.
    uptime: Time this session has been up. Format: 14 years, 51 weeks, 6 days,
      23 hours, 59 minutes, 59 seconds
    uptimeSeconds: Time this session has been up, in seconds. Format: 145
  """

    class StatusReasonValueValuesEnum(_messages.Enum):
        """Indicates why particular status was returned.

    Values:
      IPV4_PEER_ON_IPV6_ONLY_CONNECTION: BGP peer disabled because it requires
        IPv4 but the underlying connection is IPv6-only.
      IPV6_PEER_ON_IPV4_ONLY_CONNECTION: BGP peer disabled because it requires
        IPv6 but the underlying connection is IPv4-only.
      MD5_AUTH_INTERNAL_PROBLEM: Indicates internal problems with
        configuration of MD5 authentication. This particular reason can only
        be returned when md5AuthEnabled is true and status is DOWN.
      STATUS_REASON_UNSPECIFIED: <no description>
    """
        IPV4_PEER_ON_IPV6_ONLY_CONNECTION = 0
        IPV6_PEER_ON_IPV4_ONLY_CONNECTION = 1
        MD5_AUTH_INTERNAL_PROBLEM = 2
        STATUS_REASON_UNSPECIFIED = 3

    class StatusValueValuesEnum(_messages.Enum):
        """Status of the BGP peer: {UP, DOWN}

    Values:
      DOWN: <no description>
      UNKNOWN: <no description>
      UP: <no description>
    """
        DOWN = 0
        UNKNOWN = 1
        UP = 2
    advertisedRoutes = _messages.MessageField('Route', 1, repeated=True)
    bfdStatus = _messages.MessageField('BfdStatus', 2)
    enableIpv4 = _messages.BooleanField(3)
    enableIpv6 = _messages.BooleanField(4)
    ipAddress = _messages.StringField(5)
    ipv4NexthopAddress = _messages.StringField(6)
    ipv6NexthopAddress = _messages.StringField(7)
    linkedVpnTunnel = _messages.StringField(8)
    md5AuthEnabled = _messages.BooleanField(9)
    name = _messages.StringField(10)
    numLearnedRoutes = _messages.IntegerField(11, variant=_messages.Variant.UINT32)
    peerIpAddress = _messages.StringField(12)
    peerIpv4NexthopAddress = _messages.StringField(13)
    peerIpv6NexthopAddress = _messages.StringField(14)
    routerApplianceInstance = _messages.StringField(15)
    state = _messages.StringField(16)
    status = _messages.EnumField('StatusValueValuesEnum', 17)
    statusReason = _messages.EnumField('StatusReasonValueValuesEnum', 18)
    uptime = _messages.StringField(19)
    uptimeSeconds = _messages.StringField(20)