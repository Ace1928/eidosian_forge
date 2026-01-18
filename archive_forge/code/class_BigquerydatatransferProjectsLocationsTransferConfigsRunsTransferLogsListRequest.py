from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigquerydatatransferProjectsLocationsTransferConfigsRunsTransferLogsListRequest(_messages.Message):
    """A BigquerydatatransferProjectsLocationsTransferConfigsRunsTransferLogsLi
  stRequest object.

  Enums:
    MessageTypesValueValuesEnum: Message types to return. If not populated -
      INFO, WARNING and ERROR messages are returned.

  Fields:
    messageTypes: Message types to return. If not populated - INFO, WARNING
      and ERROR messages are returned.
    pageSize: Page size. The default page size is the maximum value of 1000
      results.
    pageToken: Pagination token, which can be used to request a specific page
      of `ListTransferLogsRequest` list results. For multiple-page results,
      `ListTransferLogsResponse` outputs a `next_page` token, which can be
      used as the `page_token` value to request the next page of list results.
    parent: Required. Transfer run name in the form:
      `projects/{project_id}/transferConfigs/{config_id}/runs/{run_id}` or `pr
      ojects/{project_id}/locations/{location_id}/transferConfigs/{config_id}/
      runs/{run_id}`
  """

    class MessageTypesValueValuesEnum(_messages.Enum):
        """Message types to return. If not populated - INFO, WARNING and ERROR
    messages are returned.

    Values:
      MESSAGE_SEVERITY_UNSPECIFIED: No severity specified.
      INFO: Informational message.
      WARNING: Warning message.
      ERROR: Error message.
    """
        MESSAGE_SEVERITY_UNSPECIFIED = 0
        INFO = 1
        WARNING = 2
        ERROR = 3
    messageTypes = _messages.EnumField('MessageTypesValueValuesEnum', 1, repeated=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)