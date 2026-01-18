from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsMetricDescriptorsListRequest(_messages.Message):
    """A MonitoringProjectsMetricDescriptorsListRequest object.

  Fields:
    filter: If this field is empty, all custom and system-defined metric
      descriptors are returned. Otherwise, the filter
      (https://cloud.google.com/monitoring/api/v3/filters) specifies which
      metric descriptors are to be returned. For example, the following filter
      matches all custom metrics (https://cloud.google.com/monitoring/custom-
      metrics): metric.type = starts_with("custom.googleapis.com/")
    name: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) on which to
      execute the request. The format is: projects/[PROJECT_ID_OR_NUMBER]
    pageSize: A positive number that is the maximum number of results to
      return. The default and maximum value is 10,000. If a page_size <= 0 or
      > 10,000 is submitted, will instead return a maximum of 10,000 results.
    pageToken: If this field is not empty then it must contain the
      nextPageToken value returned by a previous call to this method. Using
      this field causes the method to return additional results from the
      previous method call.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)