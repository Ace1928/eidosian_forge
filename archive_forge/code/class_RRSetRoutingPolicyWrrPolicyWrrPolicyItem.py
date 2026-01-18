from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RRSetRoutingPolicyWrrPolicyWrrPolicyItem(_messages.Message):
    """A routing block which contains the routing information for one WRR item.

  Fields:
    healthCheckedTargets: Endpoints that are health checked before making the
      routing decision. The unhealthy endpoints are omitted from the result.
      If all endpoints within a bucket are unhealthy, we choose a different
      bucket (sampled with respect to its weight) for responding. If DNSSEC is
      enabled for this zone, only one of rrdata or health_checked_targets can
      be set.
    kind: A string attribute.
    rrdatas: A string attribute.
    signatureRrdatas: DNSSEC generated signatures for all the rrdata within
      this item. Note that if health checked targets are provided for DNSSEC
      enabled zones, there's a restriction of 1 IP address per item.
    weight: The weight corresponding to this WrrPolicyItem object. When
      multiple WrrPolicyItem objects are configured, the probability of
      returning an WrrPolicyItem object's data is proportional to its weight
      relative to the sum of weights configured for all items. This weight
      must be non-negative.
  """
    healthCheckedTargets = _messages.MessageField('RRSetRoutingPolicyHealthCheckTargets', 1)
    kind = _messages.StringField(2, default='dns#rRSetRoutingPolicyWrrPolicyWrrPolicyItem')
    rrdatas = _messages.StringField(3, repeated=True)
    signatureRrdatas = _messages.StringField(4, repeated=True)
    weight = _messages.FloatField(5)