from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AreaUnderCurveParams(_messages.Message):
    """AreaUnderCurveParams groups the metrics relevant to generating duration
  based metric from base (snapshot) metric and delta (change) metric.  The
  generated metric has two dimensions:    resource usage metric and the
  duration the metric applies.  Essentially the generated metric is the Area
  Under Curve(AUC) of the "duration - resource" usage curve. This AUC metric
  is readily appliable to billing since "billable resource usage" depends on
  resource usage and duration of the resource used.  A service config may
  contain multiple resources and corresponding metrics. AreaUnderCurveParams
  groups the relevant ones: which snapshot_metric and change_metric are used
  to produce which generated_metric.

  Fields:
    changeMetric: Change of resource usage at a particular timestamp. This
      should a DELTA metric.
    generatedMetric: Metric generated from snapshot_metric and change_metric.
      This is also a DELTA metric.
    snapshotMetric: Total usage of a resource at a particular timestamp. This
      should be a GAUGE metric.
  """
    changeMetric = _messages.StringField(1)
    generatedMetric = _messages.StringField(2)
    snapshotMetric = _messages.StringField(3)