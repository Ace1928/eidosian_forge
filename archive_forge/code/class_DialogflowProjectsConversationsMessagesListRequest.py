from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationsMessagesListRequest(_messages.Message):
    """A DialogflowProjectsConversationsMessagesListRequest object.

  Fields:
    filter: Optional. Filter on message fields. Currently predicates on
      `create_time` and `create_time_epoch_microseconds` are supported.
      `create_time` only support milliseconds accuracy. E.g.,
      `create_time_epoch_microseconds > 1551790877964485` or `create_time >
      2017-01-15T01:30:15.01Z`. For more information about filtering, see [API
      Filtering](https://aip.dev/160).
    pageSize: Optional. The maximum number of items to return in a single
      page. By default 100 and at most 1000.
    pageToken: Optional. The next_page_token value returned from a previous
      list request.
    parent: Required. The name of the conversation to list messages for.
      Format: `projects//locations//conversations/`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)