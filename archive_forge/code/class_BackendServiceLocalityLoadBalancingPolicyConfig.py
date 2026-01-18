from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendServiceLocalityLoadBalancingPolicyConfig(_messages.Message):
    """Container for either a built-in LB policy supported by gRPC or Envoy or
  a custom one implemented by the end user.

  Fields:
    customPolicy: A
      BackendServiceLocalityLoadBalancingPolicyConfigCustomPolicy attribute.
    policy: A BackendServiceLocalityLoadBalancingPolicyConfigPolicy attribute.
  """
    customPolicy = _messages.MessageField('BackendServiceLocalityLoadBalancingPolicyConfigCustomPolicy', 1)
    policy = _messages.MessageField('BackendServiceLocalityLoadBalancingPolicyConfigPolicy', 2)