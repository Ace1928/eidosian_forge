from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RRSetRoutingPolicyGeoPolicyGeoPolicyItem(_messages.Message):
    """ResourceRecordSet data for one geo location.

  Fields:
    healthCheckedTargets: For A and AAAA types only. Endpoints to return in
      the query result only if they are healthy. These can be specified along
      with rrdata within this item.
    kind: A string attribute.
    location: The geo-location granularity is a GCP region. This location
      string should correspond to a GCP region. e.g. "us-east1",
      "southamerica-east1", "asia-east1", etc.
    rrdatas: A string attribute.
    signatureRrdatas: DNSSEC generated signatures for all the rrdata within
      this item. If health checked targets are provided for DNSSEC enabled
      zones, there's a restriction of 1 IP address per item.
  """
    healthCheckedTargets = _messages.MessageField('RRSetRoutingPolicyHealthCheckTargets', 1)
    kind = _messages.StringField(2, default='dns#rRSetRoutingPolicyGeoPolicyGeoPolicyItem')
    location = _messages.StringField(3)
    rrdatas = _messages.StringField(4, repeated=True)
    signatureRrdatas = _messages.StringField(5, repeated=True)