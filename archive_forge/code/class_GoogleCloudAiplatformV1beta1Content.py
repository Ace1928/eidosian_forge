from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Content(_messages.Message):
    """The base structured datatype containing multi-part content of a message.
  A `Content` includes a `role` field designating the producer of the
  `Content` and a `parts` field containing multi-part data that contains the
  content of the message turn.

  Fields:
    parts: Required. Ordered `Parts` that constitute a single message. Parts
      may have different IANA MIME types.
    role: Optional. The producer of the content. Must be either 'user' or
      'model'. Useful to set for multi-turn conversations, otherwise can be
      left blank or unset.
  """
    parts = _messages.MessageField('GoogleCloudAiplatformV1beta1Part', 1, repeated=True)
    role = _messages.StringField(2)