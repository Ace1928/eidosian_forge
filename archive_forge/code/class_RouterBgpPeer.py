from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterBgpPeer(_messages.Message):
    """A RouterBgpPeer object.

  Enums:
    AdvertiseModeValueValuesEnum: User-specified flag to indicate which mode
      to use for advertisement.
    AdvertisedGroupsValueListEntryValuesEnum:
    EnableValueValuesEnum: The status of the BGP peer connection. If set to
      FALSE, any active session with the peer is terminated and all associated
      routing information is removed. If set to TRUE, the peer connection can
      be established with routing information. The default is TRUE.
    ManagementTypeValueValuesEnum: [Output Only] The resource that configures
      and manages this BGP peer. - MANAGED_BY_USER is the default value and
      can be managed by you or other users - MANAGED_BY_ATTACHMENT is a BGP
      peer that is configured and managed by Cloud Interconnect, specifically
      by an InterconnectAttachment of type PARTNER. Google automatically
      creates, updates, and deletes this type of BGP peer when the PARTNER
      InterconnectAttachment is created, updated, or deleted.

  Fields:
    advertiseMode: User-specified flag to indicate which mode to use for
      advertisement.
    advertisedGroups: User-specified list of prefix groups to advertise in
      custom mode, which currently supports the following option: -
      ALL_SUBNETS: Advertises all of the router's own VPC subnets. This
      excludes any routes learned for subnets that use VPC Network Peering.
      Note that this field can only be populated if advertise_mode is CUSTOM
      and overrides the list defined for the router (in the "bgp" message).
      These groups are advertised in addition to any specified prefixes. Leave
      this field blank to advertise no custom groups.
    advertisedIpRanges: User-specified list of individual IP ranges to
      advertise in custom mode. This field can only be populated if
      advertise_mode is CUSTOM and overrides the list defined for the router
      (in the "bgp" message). These IP ranges are advertised in addition to
      any specified groups. Leave this field blank to advertise no custom IP
      ranges.
    advertisedRoutePriority: The priority of routes advertised to this BGP
      peer. Where there is more than one matching route of maximum length, the
      routes with the lowest priority value win.
    bfd: BFD configuration for the BGP peering.
    customLearnedIpRanges: A list of user-defined custom learned route IP
      address ranges for a BGP session.
    customLearnedRoutePriority: The user-defined custom learned route priority
      for a BGP session. This value is applied to all custom learned route
      ranges for the session. You can choose a value from `0` to `65335`. If
      you don't provide a value, Google Cloud assigns a priority of `100` to
      the ranges.
    enable: The status of the BGP peer connection. If set to FALSE, any active
      session with the peer is terminated and all associated routing
      information is removed. If set to TRUE, the peer connection can be
      established with routing information. The default is TRUE.
    enableIpv4: Enable IPv4 traffic over BGP Peer. It is enabled by default if
      the peerIpAddress is version 4.
    enableIpv6: Enable IPv6 traffic over BGP Peer. If not specified, it is
      disabled by default.
    exportPolicies: List of export policies applied to this peer, in the order
      they must be evaluated. The name must correspond to an existing policy
      that has ROUTE_POLICY_TYPE_EXPORT type.
    importPolicies: List of import policies applied to this peer, in the order
      they must be evaluated. The name must correspond to an existing policy
      that has ROUTE_POLICY_TYPE_IMPORT type.
    interfaceName: Name of the interface the BGP peer is associated with.
    ipAddress: IP address of the interface inside Google Cloud Platform. Only
      IPv4 is supported.
    ipv4NexthopAddress: IPv4 address of the interface inside Google Cloud
      Platform.
    ipv6NexthopAddress: IPv6 address of the interface inside Google Cloud
      Platform.
    managementType: [Output Only] The resource that configures and manages
      this BGP peer. - MANAGED_BY_USER is the default value and can be managed
      by you or other users - MANAGED_BY_ATTACHMENT is a BGP peer that is
      configured and managed by Cloud Interconnect, specifically by an
      InterconnectAttachment of type PARTNER. Google automatically creates,
      updates, and deletes this type of BGP peer when the PARTNER
      InterconnectAttachment is created, updated, or deleted.
    md5AuthenticationKeyName: Present if MD5 authentication is enabled for the
      peering. Must be the name of one of the entries in the
      Router.md5_authentication_keys. The field must comply with RFC1035.
    name: Name of this BGP peer. The name must be 1-63 characters long, and
      comply with RFC1035. Specifically, the name must be 1-63 characters long
      and match the regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which
      means the first character must be a lowercase letter, and all following
      characters must be a dash, lowercase letter, or digit, except the last
      character, which cannot be a dash.
    peerAsn: Peer BGP Autonomous System Number (ASN). Each BGP interface may
      use a different value.
    peerIpAddress: IP address of the BGP interface outside Google Cloud
      Platform. Only IPv4 is supported.
    peerIpv4NexthopAddress: IPv4 address of the BGP interface outside Google
      Cloud Platform.
    peerIpv6NexthopAddress: IPv6 address of the BGP interface outside Google
      Cloud Platform.
    routerApplianceInstance: URI of the VM instance that is used as third-
      party router appliances such as Next Gen Firewalls, Virtual Routers, or
      Router Appliances. The VM instance must be located in zones contained in
      the same region as this Cloud Router. The VM instance is the peer side
      of the BGP session.
  """

    class AdvertiseModeValueValuesEnum(_messages.Enum):
        """User-specified flag to indicate which mode to use for advertisement.

    Values:
      CUSTOM: <no description>
      DEFAULT: <no description>
    """
        CUSTOM = 0
        DEFAULT = 1

    class AdvertisedGroupsValueListEntryValuesEnum(_messages.Enum):
        """AdvertisedGroupsValueListEntryValuesEnum enum type.

    Values:
      ALL_SUBNETS: Advertise all available subnets (including peer VPC
        subnets).
    """
        ALL_SUBNETS = 0

    class EnableValueValuesEnum(_messages.Enum):
        """The status of the BGP peer connection. If set to FALSE, any active
    session with the peer is terminated and all associated routing information
    is removed. If set to TRUE, the peer connection can be established with
    routing information. The default is TRUE.

    Values:
      FALSE: <no description>
      TRUE: <no description>
    """
        FALSE = 0
        TRUE = 1

    class ManagementTypeValueValuesEnum(_messages.Enum):
        """[Output Only] The resource that configures and manages this BGP peer.
    - MANAGED_BY_USER is the default value and can be managed by you or other
    users - MANAGED_BY_ATTACHMENT is a BGP peer that is configured and managed
    by Cloud Interconnect, specifically by an InterconnectAttachment of type
    PARTNER. Google automatically creates, updates, and deletes this type of
    BGP peer when the PARTNER InterconnectAttachment is created, updated, or
    deleted.

    Values:
      MANAGED_BY_ATTACHMENT: The BGP peer is automatically created for PARTNER
        type InterconnectAttachment; Google will automatically create/delete
        this BGP peer when the PARTNER InterconnectAttachment is
        created/deleted, and Google will update the ipAddress and
        peerIpAddress when the PARTNER InterconnectAttachment is provisioned.
        This type of BGP peer cannot be created or deleted, but can be
        modified for all fields except for name, ipAddress and peerIpAddress.
      MANAGED_BY_USER: Default value, the BGP peer is manually created and
        managed by user.
    """
        MANAGED_BY_ATTACHMENT = 0
        MANAGED_BY_USER = 1
    advertiseMode = _messages.EnumField('AdvertiseModeValueValuesEnum', 1)
    advertisedGroups = _messages.EnumField('AdvertisedGroupsValueListEntryValuesEnum', 2, repeated=True)
    advertisedIpRanges = _messages.MessageField('RouterAdvertisedIpRange', 3, repeated=True)
    advertisedRoutePriority = _messages.IntegerField(4, variant=_messages.Variant.UINT32)
    bfd = _messages.MessageField('RouterBgpPeerBfd', 5)
    customLearnedIpRanges = _messages.MessageField('RouterBgpPeerCustomLearnedIpRange', 6, repeated=True)
    customLearnedRoutePriority = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    enable = _messages.EnumField('EnableValueValuesEnum', 8)
    enableIpv4 = _messages.BooleanField(9)
    enableIpv6 = _messages.BooleanField(10)
    exportPolicies = _messages.StringField(11, repeated=True)
    importPolicies = _messages.StringField(12, repeated=True)
    interfaceName = _messages.StringField(13)
    ipAddress = _messages.StringField(14)
    ipv4NexthopAddress = _messages.StringField(15)
    ipv6NexthopAddress = _messages.StringField(16)
    managementType = _messages.EnumField('ManagementTypeValueValuesEnum', 17)
    md5AuthenticationKeyName = _messages.StringField(18)
    name = _messages.StringField(19)
    peerAsn = _messages.IntegerField(20, variant=_messages.Variant.UINT32)
    peerIpAddress = _messages.StringField(21)
    peerIpv4NexthopAddress = _messages.StringField(22)
    peerIpv6NexthopAddress = _messages.StringField(23)
    routerApplianceInstance = _messages.StringField(24)