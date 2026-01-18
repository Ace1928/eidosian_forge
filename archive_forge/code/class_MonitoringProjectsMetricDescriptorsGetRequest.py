from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsMetricDescriptorsGetRequest(_messages.Message):
    """A MonitoringProjectsMetricDescriptorsGetRequest object.

  Fields:
    name: Required. The metric descriptor on which to execute the request. The
      format is: projects/[PROJECT_ID_OR_NUMBER]/metricDescriptors/[METRIC_ID]
      An example value of [METRIC_ID] is
      "compute.googleapis.com/instance/disk/read_bytes_count".
  """
    name = _messages.StringField(1, required=True)