from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerStepNamespaceMetrics(_messages.Message):
    """Metrics for a particular unfused step and namespace. A metric is
  uniquely identified by the `metrics_namespace`, `original_step`, `metric
  name` and `metric_labels`.

  Fields:
    metricValues: Optional. Metrics that are recorded for this namespace and
      unfused step.
    metricsNamespace: The namespace of these metrics on the worker.
    originalStep: The original system name of the unfused step that these
      metrics are reported from.
  """
    metricValues = _messages.MessageField('MetricValue', 1, repeated=True)
    metricsNamespace = _messages.StringField(2)
    originalStep = _messages.StringField(3)