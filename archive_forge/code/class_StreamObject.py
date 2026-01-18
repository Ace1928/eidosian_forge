from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamObject(_messages.Message):
    """A specific stream object (e.g a specific DB table).

  Fields:
    backfillJob: The latest backfill job that was initiated for the stream
      object.
    createTime: Output only. The creation time of the object.
    displayName: Required. Display name.
    errors: Output only. Active errors on the object.
    name: Output only. The object resource's name.
    sourceObject: The object identifier in the data source.
    updateTime: Output only. The last update time of the object.
  """
    backfillJob = _messages.MessageField('BackfillJob', 1)
    createTime = _messages.StringField(2)
    displayName = _messages.StringField(3)
    errors = _messages.MessageField('Error', 4, repeated=True)
    name = _messages.StringField(5)
    sourceObject = _messages.MessageField('SourceObjectIdentifier', 6)
    updateTime = _messages.StringField(7)