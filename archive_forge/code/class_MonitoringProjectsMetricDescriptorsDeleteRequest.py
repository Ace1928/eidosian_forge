from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsMetricDescriptorsDeleteRequest(_messages.Message):
    """A MonitoringProjectsMetricDescriptorsDeleteRequest object.

  Fields:
    name: Required. The metric descriptor on which to execute the request. The
      format is: projects/[PROJECT_ID_OR_NUMBER]/metricDescriptors/[METRIC_ID]
      An example of [METRIC_ID] is: "custom.googleapis.com/my_test_metric".
  """
    name = _messages.StringField(1, required=True)