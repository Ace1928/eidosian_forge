from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminPortConfig(_messages.Message):
    """BareMetalAdminPortConfig is the specification of load balancer ports.

  Fields:
    controlPlaneLoadBalancerPort: The port that control plane hosted load
      balancers will listen on.
  """
    controlPlaneLoadBalancerPort = _messages.IntegerField(1, variant=_messages.Variant.INT32)