from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListKeyValueEntriesResponse(_messages.Message):
    """The request structure for listing key value map keys and its
  corresponding values.

  Fields:
    keyValueEntries: One or more key value map keys and values.
    nextPageToken: Token that can be sent as `next_page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    keyValueEntries = _messages.MessageField('GoogleCloudApigeeV1KeyValueEntry', 1, repeated=True)
    nextPageToken = _messages.StringField(2)