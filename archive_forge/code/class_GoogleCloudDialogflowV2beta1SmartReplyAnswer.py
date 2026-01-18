from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1SmartReplyAnswer(_messages.Message):
    """Represents a smart reply answer.

  Fields:
    answerRecord: The name of answer record, in the format of
      "projects//locations//answerRecords/"
    confidence: Smart reply confidence. The system's confidence score that
      this reply is a good match for this conversation, as a value from 0.0
      (completely uncertain) to 1.0 (completely certain).
    reply: The content of the reply.
  """
    answerRecord = _messages.StringField(1)
    confidence = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    reply = _messages.StringField(3)