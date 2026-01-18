from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneBgpLbConfig(_messages.Message):
    """BareMetalStandaloneBgpLbConfig represents configuration parameters for a
  Border Gateway Protocol (BGP) load balancer.

  Fields:
    addressPools: Required. AddressPools is a list of non-overlapping IP pools
      used by load balancer typed services. All addresses must be routable to
      load balancer nodes. IngressVIP must be included in the pools.
    asn: Required. BGP autonomous system number (ASN) of the cluster. This
      field can be updated after cluster creation.
    bgpPeerConfigs: Required. The list of BGP peers that the cluster will
      connect to. At least one peer must be configured for each control plane
      node. Control plane nodes will connect to these peers to advertise the
      control plane VIP. The Services load balancer also uses these peers by
      default. This field can be updated after cluster creation.
    loadBalancerNodePoolConfig: Specifies the node pool running data plane
      load balancing. L2 connectivity is required among nodes in this pool. If
      missing, the control plane node pool is used for data plane load
      balancing.
  """
    addressPools = _messages.MessageField('BareMetalStandaloneLoadBalancerAddressPool', 1, repeated=True)
    asn = _messages.IntegerField(2)
    bgpPeerConfigs = _messages.MessageField('BareMetalStandaloneBgpPeerConfig', 3, repeated=True)
    loadBalancerNodePoolConfig = _messages.MessageField('BareMetalStandaloneLoadBalancerNodePoolConfig', 4)