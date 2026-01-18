from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WriteResult(_messages.Message):
    """The result of applying a write.

  Fields:
    transformResults: The results of applying each
      DocumentTransform.FieldTransform, in the same order.
    updateTime: The last update time of the document after applying the write.
      Not set after a `delete`. If the write did not actually change the
      document, this will be the previous update_time.
  """
    transformResults = _messages.MessageField('Value', 1, repeated=True)
    updateTime = _messages.StringField(2)