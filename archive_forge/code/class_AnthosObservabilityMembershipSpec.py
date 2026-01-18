from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnthosObservabilityMembershipSpec(_messages.Message):
    """**Anthosobservability**: Per-Membership Feature spec.

  Fields:
    doNotOptimizeMetrics: Use full of metrics rather than optimized metrics.
      See https://cloud.google.com/anthos/clusters/docs/on-
      prem/1.8/concepts/logging-and-
      monitoring#optimized_metrics_default_metrics
    enableStackdriverOnApplications: Enable collecting and reporting metrics
      and logs from user apps.
    version: the version of stackdriver operator used by this feature
  """
    doNotOptimizeMetrics = _messages.BooleanField(1)
    enableStackdriverOnApplications = _messages.BooleanField(2)
    version = _messages.StringField(3)