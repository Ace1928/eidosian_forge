from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AsyncQueryResultView(_messages.Message):
    """A GoogleCloudApigeeV1AsyncQueryResultView object.

  Fields:
    code: Error code when there is a failure.
    error: Error message when there is a failure.
    metadata: Metadata contains information like metrics, dimenstions etc of
      the AsyncQuery.
    rows: Rows of query result. Each row is a JSON object. Example:
      {sum(message_count): 1, developer_app: "(not set)",...}
    state: State of retrieving ResultView.
  """
    code = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    error = _messages.StringField(2)
    metadata = _messages.MessageField('GoogleCloudApigeeV1QueryMetadata', 3)
    rows = _messages.MessageField('extra_types.JsonValue', 4, repeated=True)
    state = _messages.StringField(5)