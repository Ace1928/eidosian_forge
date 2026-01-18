from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareMetalLbConfig(_messages.Message):
    """Represents configuration parameters for the MetalLB load balancer.

  Fields:
    addressPools: Required. AddressPools is a list of non-overlapping IP pools
      used by load balancer typed services. All addresses must be routable to
      load balancer nodes. IngressVIP must be included in the pools.
  """
    addressPools = _messages.MessageField('VmwareAddressPool', 1, repeated=True)