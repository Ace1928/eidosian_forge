from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsMonitoredResourceDescriptorsListRequest(_messages.Message):
    """A MonitoringProjectsMonitoredResourceDescriptorsListRequest object.

  Fields:
    filter: An optional filter
      (https://cloud.google.com/monitoring/api/v3/filters) describing the
      descriptors to be returned. The filter can reference the descriptor's
      type and labels. For example, the following filter returns only Google
      Compute Engine descriptors that have an id label: resource.type =
      starts_with("gce_") AND resource.label:id
    name: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) on which to
      execute the request. The format is: projects/[PROJECT_ID_OR_NUMBER]
    pageSize: A positive number that is the maximum number of results to
      return.
    pageToken: If this field is not empty then it must contain the
      nextPageToken value returned by a previous call to this method. Using
      this field causes the method to return additional results from the
      previous method call.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)