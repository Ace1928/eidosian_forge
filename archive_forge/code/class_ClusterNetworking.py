from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterNetworking(_messages.Message):
    """Cluster-wide networking configuration.

  Enums:
    NetworkTypeValueValuesEnum: Output only. IP addressing type of this
      cluster i.e. SINGLESTACK_V4 vs DUALSTACK_V4_V6

  Fields:
    clusterIpv4CidrBlocks: Required. All pods in the cluster are assigned an
      RFC1918 IPv4 address from these blocks. Only a single block is
      supported. This field cannot be changed after creation.
    clusterIpv6CidrBlocks: Immutable. If specified, dual stack mode is enabled
      and all pods in the cluster are assigned an IPv6 address from these
      blocks alongside from an IPv4 address. Only a single block is supported.
      This field cannot be changed after creation.
    networkType: Output only. IP addressing type of this cluster i.e.
      SINGLESTACK_V4 vs DUALSTACK_V4_V6
    servicesIpv4CidrBlocks: Required. All services in the cluster are assigned
      an RFC1918 IPv4 address from these blocks. Only a single block is
      supported. This field cannot be changed after creation.
    servicesIpv6CidrBlocks: Immutable. If specified, dual stack mode is
      enabled and all services in the cluster are assigned an IPv6 address
      from these blocks alongside from an IPv4 address. Only a single block is
      supported. This field cannot be changed after creation.
  """

    class NetworkTypeValueValuesEnum(_messages.Enum):
        """Output only. IP addressing type of this cluster i.e. SINGLESTACK_V4 vs
    DUALSTACK_V4_V6

    Values:
      NETWORK_TYPE_UNSPECIFIED: Unknown cluster type
      SINGLESTACK_V4: SingleStack v4 address only cluster
      DUALSTACK_V4_V6: DualStack cluster - support v4 and v6 address
    """
        NETWORK_TYPE_UNSPECIFIED = 0
        SINGLESTACK_V4 = 1
        DUALSTACK_V4_V6 = 2
    clusterIpv4CidrBlocks = _messages.StringField(1, repeated=True)
    clusterIpv6CidrBlocks = _messages.StringField(2, repeated=True)
    networkType = _messages.EnumField('NetworkTypeValueValuesEnum', 3)
    servicesIpv4CidrBlocks = _messages.StringField(4, repeated=True)
    servicesIpv6CidrBlocks = _messages.StringField(5, repeated=True)