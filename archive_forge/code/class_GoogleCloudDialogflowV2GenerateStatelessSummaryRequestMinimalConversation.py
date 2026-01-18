from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2GenerateStatelessSummaryRequestMinimalConversation(_messages.Message):
    """The minimum amount of information required to generate a Summary without
  having a Conversation resource created.

  Fields:
    messages: Required. The messages that the Summary will be generated from.
      It is expected that this message content is already redacted and does
      not contain any PII. Required fields: {content, language_code,
      participant, participant_role} Optional fields: {send_time} If send_time
      is not provided, then the messages must be provided in chronological
      order.
  """
    messages = _messages.MessageField('GoogleCloudDialogflowV2Message', 1, repeated=True)