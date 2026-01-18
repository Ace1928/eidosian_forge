from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalPortConfig(_messages.Message):
    """Specifies load balancer ports for the bare metal user cluster.

  Fields:
    controlPlaneLoadBalancerPort: The port that control plane hosted load
      balancers will listen on.
  """
    controlPlaneLoadBalancerPort = _messages.IntegerField(1, variant=_messages.Variant.INT32)