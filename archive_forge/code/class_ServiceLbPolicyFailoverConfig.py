from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceLbPolicyFailoverConfig(_messages.Message):
    """Option to specify health based failover behavior. This is not related to
  Network load balancer FailoverPolicy.

  Fields:
    failoverHealthThreshold: Optional. The percentage threshold that a load
      balancer will begin to send traffic to failover backends. If the
      percentage of endpoints in a MIG/NEG is smaller than this value, traffic
      would be sent to failover backends if possible. This field should be set
      to a value between 1 and 99. The default value is 50 for Global external
      HTTP(S) load balancer (classic) and Proxyless service mesh, and 70 for
      others.
  """
    failoverHealthThreshold = _messages.IntegerField(1, variant=_messages.Variant.INT32)