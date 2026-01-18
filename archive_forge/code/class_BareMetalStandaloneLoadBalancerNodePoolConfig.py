from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneLoadBalancerNodePoolConfig(_messages.Message):
    """Specifies the load balancer's node pool configuration.

  Fields:
    nodePoolConfig: The generic configuration for a node pool running a load
      balancer.
  """
    nodePoolConfig = _messages.MessageField('BareMetalNodePoolConfig', 1)