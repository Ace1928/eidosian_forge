from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsNotificationChannelsListRequest(_messages.Message):
    """A MonitoringProjectsNotificationChannelsListRequest object.

  Fields:
    filter: If provided, this field specifies the criteria that must be met by
      notification channels to be included in the response.For more details,
      see sorting and filtering
      (https://cloud.google.com/monitoring/api/v3/sorting-and-filtering).
    name: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) on which to
      execute the request. The format is: projects/[PROJECT_ID_OR_NUMBER] This
      names the container in which to look for the notification channels; it
      does not name a specific channel. To query a specific channel by REST
      resource name, use the GetNotificationChannel operation.
    orderBy: A comma-separated list of fields by which to sort the result.
      Supports the same set of fields as in filter. Entries can be prefixed
      with a minus sign to sort in descending rather than ascending order.For
      more details, see sorting and filtering
      (https://cloud.google.com/monitoring/api/v3/sorting-and-filtering).
    pageSize: The maximum number of results to return in a single response. If
      not set to a positive number, a reasonable value will be chosen by the
      service.
    pageToken: If non-empty, page_token must contain a value returned as the
      next_page_token in a previous response to request the next set of
      results.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    orderBy = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)