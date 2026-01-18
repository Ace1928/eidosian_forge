from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalMetalLbConfig(_messages.Message):
    """Represents configuration parameters for a MetalLB load balancer.

  Fields:
    addressPools: Required. AddressPools is a list of non-overlapping IP pools
      used by load balancer typed services. All addresses must be routable to
      load balancer nodes. IngressVIP must be included in the pools.
    loadBalancerNodePoolConfig: Specifies the node pool running the load
      balancer. L2 connectivity is required among nodes in this pool. If
      missing, the control plane node pool is used as the load balancer pool.
  """
    addressPools = _messages.MessageField('BareMetalLoadBalancerAddressPool', 1, repeated=True)
    loadBalancerNodePoolConfig = _messages.MessageField('BareMetalLoadBalancerNodePoolConfig', 2)