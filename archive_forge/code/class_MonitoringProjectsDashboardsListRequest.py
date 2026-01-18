from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsDashboardsListRequest(_messages.Message):
    """A MonitoringProjectsDashboardsListRequest object.

  Fields:
    pageSize: A positive number that is the maximum number of results to
      return. If unspecified, a default of 1000 is used.
    pageToken: Optional. If this field is not empty then it must contain the
      nextPageToken value returned by a previous call to this method. Using
      this field causes the method to return additional results from the
      previous method call.
    parent: Required. The scope of the dashboards to list. The format is:
      projects/[PROJECT_ID_OR_NUMBER]
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)