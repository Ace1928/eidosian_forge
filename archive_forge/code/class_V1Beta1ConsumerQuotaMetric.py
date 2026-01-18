from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V1Beta1ConsumerQuotaMetric(_messages.Message):
    """Consumer quota settings for a quota metric.

  Fields:
    consumerQuotaLimits: The consumer quota for each quota limit defined on
      the metric.
    displayName: The display name of the metric.  An example name would be:
      "CPUs"
    metric: The name of the metric.  An example name would be:
      `compute.googleapis.com/cpus`
    name: The resource name of the quota settings on this metric for this
      consumer.  An example name would be: `services/serviceconsumermanagement
      .googleapis.com/projects/123/quota/metrics/compute.googleapis.com%2Fcpus
      The resource name is intended to be opaque and should not be parsed for
      its component strings, since its representation could change in the
      future.
    unit: The units in which the metric value is reported.
  """
    consumerQuotaLimits = _messages.MessageField('V1Beta1ConsumerQuotaLimit', 1, repeated=True)
    displayName = _messages.StringField(2)
    metric = _messages.StringField(3)
    name = _messages.StringField(4)
    unit = _messages.StringField(5)