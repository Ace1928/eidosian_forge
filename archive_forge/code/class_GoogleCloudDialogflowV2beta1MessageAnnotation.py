from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1MessageAnnotation(_messages.Message):
    """Represents the result of annotation for the message.

  Fields:
    containEntities: Required. Indicates whether the text message contains
      entities.
    parts: Optional. The collection of annotated message parts ordered by
      their position in the message. You can recover the annotated message by
      concatenating [AnnotatedMessagePart.text].
  """
    containEntities = _messages.BooleanField(1)
    parts = _messages.MessageField('GoogleCloudDialogflowV2beta1AnnotatedMessagePart', 2, repeated=True)