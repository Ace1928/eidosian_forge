from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsAlertPoliciesListRequest(_messages.Message):
    """A MonitoringProjectsAlertPoliciesListRequest object.

  Fields:
    filter: If provided, this field specifies the criteria that must be met by
      alert policies to be included in the response.For more details, see
      sorting and filtering
      (https://cloud.google.com/monitoring/api/v3/sorting-and-filtering).
    name: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) whose alert
      policies are to be listed. The format is:
      projects/[PROJECT_ID_OR_NUMBER] Note that this field names the parent
      container in which the alerting policies to be listed are stored. To
      retrieve a single alerting policy by name, use the GetAlertPolicy
      operation, instead.
    orderBy: A comma-separated list of fields by which to sort the result.
      Supports the same set of field references as the filter field. Entries
      can be prefixed with a minus sign to sort by the field in descending
      order.For more details, see sorting and filtering
      (https://cloud.google.com/monitoring/api/v3/sorting-and-filtering).
    pageSize: The maximum number of results to return in a single response.
    pageToken: If this field is not empty then it must contain the
      nextPageToken value returned by a previous call to this method. Using
      this field causes the method to return more results from the previous
      method call.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    orderBy = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)