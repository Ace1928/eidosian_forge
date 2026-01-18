from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingProjectsMetricsDeleteRequest(_messages.Message):
    """A LoggingProjectsMetricsDeleteRequest object.

  Fields:
    metricName: Required. The resource name of the metric to delete:
      "projects/[PROJECT_ID]/metrics/[METRIC_ID]"
  """
    metricName = _messages.StringField(1, required=True)