from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkPlacementNetworkFeatures(_messages.Message):
    """A NetworkPlacementNetworkFeatures object.

  Enums:
    AllowAutoModeSubnetValueValuesEnum: Specifies whether auto mode subnet
      creation is allowed.
    AllowCloudNatValueValuesEnum: Specifies whether cloud NAT creation is
      allowed.
    AllowCloudRouterValueValuesEnum: Specifies whether cloud router creation
      is allowed.
    AllowInterconnectValueValuesEnum: Specifies whether Cloud Interconnect
      creation is allowed.
    AllowLoadBalancingValueValuesEnum: Specifies whether cloud load balancing
      is allowed.
    AllowMultiNicInSameNetworkValueValuesEnum: Specifies whether multi-nic in
      the same network is allowed.
    AllowPacketMirroringValueValuesEnum: Specifies whether Packet Mirroring
      1.0 is supported.
    AllowPrivateGoogleAccessValueValuesEnum: Specifies whether private Google
      access is allowed.
    AllowPscValueValuesEnum: Specifies whether PSC creation is allowed.
    AllowSameNetworkUnicastValueValuesEnum: Specifies whether unicast within
      the same network is allowed.
    AllowStaticRoutesValueValuesEnum: Specifies whether static route creation
      is allowed.
    AllowVpcPeeringValueValuesEnum: Specifies whether VPC peering is allowed.
    AllowVpnValueValuesEnum: Specifies whether VPN creation is allowed.
    AllowedSubnetPurposesValueListEntryValuesEnum:
    AllowedSubnetStackTypesValueListEntryValuesEnum:
    InterfaceTypesValueListEntryValuesEnum:
    MulticastValueValuesEnum: Specifies which type of multicast is supported.
    UnicastValueValuesEnum: Specifies which type of unicast is supported.

  Fields:
    allowAutoModeSubnet: Specifies whether auto mode subnet creation is
      allowed.
    allowCloudNat: Specifies whether cloud NAT creation is allowed.
    allowCloudRouter: Specifies whether cloud router creation is allowed.
    allowInterconnect: Specifies whether Cloud Interconnect creation is
      allowed.
    allowLoadBalancing: Specifies whether cloud load balancing is allowed.
    allowMultiNicInSameNetwork: Specifies whether multi-nic in the same
      network is allowed.
    allowPacketMirroring: Specifies whether Packet Mirroring 1.0 is supported.
    allowPrivateGoogleAccess: Specifies whether private Google access is
      allowed.
    allowPsc: Specifies whether PSC creation is allowed.
    allowSameNetworkUnicast: Specifies whether unicast within the same network
      is allowed.
    allowStaticRoutes: Specifies whether static route creation is allowed.
    allowVpcPeering: Specifies whether VPC peering is allowed.
    allowVpn: Specifies whether VPN creation is allowed.
    allowedSubnetPurposes: Specifies which subnetwork purposes are supported.
    allowedSubnetStackTypes: Specifies which subnetwork stack types are
      supported.
    interfaceTypes: If set, limits the interface types that the network
      supports. If empty, all interface types are supported.
    multicast: Specifies which type of multicast is supported.
    unicast: Specifies which type of unicast is supported.
  """

    class AllowAutoModeSubnetValueValuesEnum(_messages.Enum):
        """Specifies whether auto mode subnet creation is allowed.

    Values:
      AUTO_MODE_SUBNET_ALLOWED: <no description>
      AUTO_MODE_SUBNET_BLOCKED: <no description>
      AUTO_MODE_SUBNET_UNSPECIFIED: <no description>
    """
        AUTO_MODE_SUBNET_ALLOWED = 0
        AUTO_MODE_SUBNET_BLOCKED = 1
        AUTO_MODE_SUBNET_UNSPECIFIED = 2

    class AllowCloudNatValueValuesEnum(_messages.Enum):
        """Specifies whether cloud NAT creation is allowed.

    Values:
      CLOUD_NAT_ALLOWED: <no description>
      CLOUD_NAT_BLOCKED: <no description>
      CLOUD_NAT_UNSPECIFIED: <no description>
    """
        CLOUD_NAT_ALLOWED = 0
        CLOUD_NAT_BLOCKED = 1
        CLOUD_NAT_UNSPECIFIED = 2

    class AllowCloudRouterValueValuesEnum(_messages.Enum):
        """Specifies whether cloud router creation is allowed.

    Values:
      CLOUD_ROUTER_ALLOWED: <no description>
      CLOUD_ROUTER_BLOCKED: <no description>
      CLOUD_ROUTER_UNSPECIFIED: <no description>
    """
        CLOUD_ROUTER_ALLOWED = 0
        CLOUD_ROUTER_BLOCKED = 1
        CLOUD_ROUTER_UNSPECIFIED = 2

    class AllowInterconnectValueValuesEnum(_messages.Enum):
        """Specifies whether Cloud Interconnect creation is allowed.

    Values:
      INTERCONNECT_ALLOWED: <no description>
      INTERCONNECT_BLOCKED: <no description>
      INTERCONNECT_UNSPECIFIED: <no description>
    """
        INTERCONNECT_ALLOWED = 0
        INTERCONNECT_BLOCKED = 1
        INTERCONNECT_UNSPECIFIED = 2

    class AllowLoadBalancingValueValuesEnum(_messages.Enum):
        """Specifies whether cloud load balancing is allowed.

    Values:
      LOAD_BALANCING_ALLOWED: <no description>
      LOAD_BALANCING_BLOCKED: <no description>
      LOAD_BALANCING_UNSPECIFIED: <no description>
    """
        LOAD_BALANCING_ALLOWED = 0
        LOAD_BALANCING_BLOCKED = 1
        LOAD_BALANCING_UNSPECIFIED = 2

    class AllowMultiNicInSameNetworkValueValuesEnum(_messages.Enum):
        """Specifies whether multi-nic in the same network is allowed.

    Values:
      MULTI_NIC_IN_SAME_NETWORK_ALLOWED: <no description>
      MULTI_NIC_IN_SAME_NETWORK_BLOCKED: <no description>
      MULTI_NIC_IN_SAME_NETWORK_UNSPECIFIED: <no description>
    """
        MULTI_NIC_IN_SAME_NETWORK_ALLOWED = 0
        MULTI_NIC_IN_SAME_NETWORK_BLOCKED = 1
        MULTI_NIC_IN_SAME_NETWORK_UNSPECIFIED = 2

    class AllowPacketMirroringValueValuesEnum(_messages.Enum):
        """Specifies whether Packet Mirroring 1.0 is supported.

    Values:
      PACKET_MIRRORING_ALLOWED: <no description>
      PACKET_MIRRORING_BLOCKED: <no description>
      PACKET_MIRRORING_UNSPECIFIED: <no description>
    """
        PACKET_MIRRORING_ALLOWED = 0
        PACKET_MIRRORING_BLOCKED = 1
        PACKET_MIRRORING_UNSPECIFIED = 2

    class AllowPrivateGoogleAccessValueValuesEnum(_messages.Enum):
        """Specifies whether private Google access is allowed.

    Values:
      PRIVATE_GOOGLE_ACCESS_ALLOWED: <no description>
      PRIVATE_GOOGLE_ACCESS_BLOCKED: <no description>
      PRIVATE_GOOGLE_ACCESS_UNSPECIFIED: <no description>
    """
        PRIVATE_GOOGLE_ACCESS_ALLOWED = 0
        PRIVATE_GOOGLE_ACCESS_BLOCKED = 1
        PRIVATE_GOOGLE_ACCESS_UNSPECIFIED = 2

    class AllowPscValueValuesEnum(_messages.Enum):
        """Specifies whether PSC creation is allowed.

    Values:
      PSC_ALLOWED: <no description>
      PSC_BLOCKED: <no description>
      PSC_UNSPECIFIED: <no description>
    """
        PSC_ALLOWED = 0
        PSC_BLOCKED = 1
        PSC_UNSPECIFIED = 2

    class AllowSameNetworkUnicastValueValuesEnum(_messages.Enum):
        """Specifies whether unicast within the same network is allowed.

    Values:
      SAME_NETWORK_UNICAST_ALLOWED: <no description>
      SAME_NETWORK_UNICAST_BLOCKED: <no description>
      SAME_NETWORK_UNICAST_UNSPECIFIED: <no description>
    """
        SAME_NETWORK_UNICAST_ALLOWED = 0
        SAME_NETWORK_UNICAST_BLOCKED = 1
        SAME_NETWORK_UNICAST_UNSPECIFIED = 2

    class AllowStaticRoutesValueValuesEnum(_messages.Enum):
        """Specifies whether static route creation is allowed.

    Values:
      STATIC_ROUTES_ALLOWED: <no description>
      STATIC_ROUTES_BLOCKED: <no description>
      STATIC_ROUTES_UNSPECIFIED: <no description>
    """
        STATIC_ROUTES_ALLOWED = 0
        STATIC_ROUTES_BLOCKED = 1
        STATIC_ROUTES_UNSPECIFIED = 2

    class AllowVpcPeeringValueValuesEnum(_messages.Enum):
        """Specifies whether VPC peering is allowed.

    Values:
      VPC_PEERING_ALLOWED: <no description>
      VPC_PEERING_BLOCKED: <no description>
      VPC_PEERING_UNSPECIFIED: <no description>
    """
        VPC_PEERING_ALLOWED = 0
        VPC_PEERING_BLOCKED = 1
        VPC_PEERING_UNSPECIFIED = 2

    class AllowVpnValueValuesEnum(_messages.Enum):
        """Specifies whether VPN creation is allowed.

    Values:
      VPN_ALLOWED: <no description>
      VPN_BLOCKED: <no description>
      VPN_UNSPECIFIED: <no description>
    """
        VPN_ALLOWED = 0
        VPN_BLOCKED = 1
        VPN_UNSPECIFIED = 2

    class AllowedSubnetPurposesValueListEntryValuesEnum(_messages.Enum):
        """AllowedSubnetPurposesValueListEntryValuesEnum enum type.

    Values:
      SUBNET_PURPOSE_CUSTOM_HARDWARE: <no description>
      SUBNET_PURPOSE_PRIVATE: <no description>
      SUBNET_PURPOSE_UNSPECIFIED: <no description>
    """
        SUBNET_PURPOSE_CUSTOM_HARDWARE = 0
        SUBNET_PURPOSE_PRIVATE = 1
        SUBNET_PURPOSE_UNSPECIFIED = 2

    class AllowedSubnetStackTypesValueListEntryValuesEnum(_messages.Enum):
        """AllowedSubnetStackTypesValueListEntryValuesEnum enum type.

    Values:
      SUBNET_STACK_TYPE_IPV4_IPV6: <no description>
      SUBNET_STACK_TYPE_IPV4_ONLY: <no description>
      SUBNET_STACK_TYPE_IPV6_ONLY: <no description>
      SUBNET_STACK_TYPE_UNSPECIFIED: <no description>
    """
        SUBNET_STACK_TYPE_IPV4_IPV6 = 0
        SUBNET_STACK_TYPE_IPV4_ONLY = 1
        SUBNET_STACK_TYPE_IPV6_ONLY = 2
        SUBNET_STACK_TYPE_UNSPECIFIED = 3

    class InterfaceTypesValueListEntryValuesEnum(_messages.Enum):
        """InterfaceTypesValueListEntryValuesEnum enum type.

    Values:
      GVNIC: GVNIC
      IDPF: IDPF
      RDMA: DEPRECATED: Please use TNA_IRDMA instead.
      UNSPECIFIED_NIC_TYPE: No type specified.
      VIRTIO_NET: VIRTIO
    """
        GVNIC = 0
        IDPF = 1
        RDMA = 2
        UNSPECIFIED_NIC_TYPE = 3
        VIRTIO_NET = 4

    class MulticastValueValuesEnum(_messages.Enum):
        """Specifies which type of multicast is supported.

    Values:
      MULTICAST_SDN: <no description>
      MULTICAST_ULL: <no description>
      MULTICAST_UNSPECIFIED: <no description>
    """
        MULTICAST_SDN = 0
        MULTICAST_ULL = 1
        MULTICAST_UNSPECIFIED = 2

    class UnicastValueValuesEnum(_messages.Enum):
        """Specifies which type of unicast is supported.

    Values:
      UNICAST_SDN: <no description>
      UNICAST_ULL: <no description>
      UNICAST_UNSPECIFIED: <no description>
    """
        UNICAST_SDN = 0
        UNICAST_ULL = 1
        UNICAST_UNSPECIFIED = 2
    allowAutoModeSubnet = _messages.EnumField('AllowAutoModeSubnetValueValuesEnum', 1)
    allowCloudNat = _messages.EnumField('AllowCloudNatValueValuesEnum', 2)
    allowCloudRouter = _messages.EnumField('AllowCloudRouterValueValuesEnum', 3)
    allowInterconnect = _messages.EnumField('AllowInterconnectValueValuesEnum', 4)
    allowLoadBalancing = _messages.EnumField('AllowLoadBalancingValueValuesEnum', 5)
    allowMultiNicInSameNetwork = _messages.EnumField('AllowMultiNicInSameNetworkValueValuesEnum', 6)
    allowPacketMirroring = _messages.EnumField('AllowPacketMirroringValueValuesEnum', 7)
    allowPrivateGoogleAccess = _messages.EnumField('AllowPrivateGoogleAccessValueValuesEnum', 8)
    allowPsc = _messages.EnumField('AllowPscValueValuesEnum', 9)
    allowSameNetworkUnicast = _messages.EnumField('AllowSameNetworkUnicastValueValuesEnum', 10)
    allowStaticRoutes = _messages.EnumField('AllowStaticRoutesValueValuesEnum', 11)
    allowVpcPeering = _messages.EnumField('AllowVpcPeeringValueValuesEnum', 12)
    allowVpn = _messages.EnumField('AllowVpnValueValuesEnum', 13)
    allowedSubnetPurposes = _messages.EnumField('AllowedSubnetPurposesValueListEntryValuesEnum', 14, repeated=True)
    allowedSubnetStackTypes = _messages.EnumField('AllowedSubnetStackTypesValueListEntryValuesEnum', 15, repeated=True)
    interfaceTypes = _messages.EnumField('InterfaceTypesValueListEntryValuesEnum', 16, repeated=True)
    multicast = _messages.EnumField('MulticastValueValuesEnum', 17)
    unicast = _messages.EnumField('UnicastValueValuesEnum', 18)