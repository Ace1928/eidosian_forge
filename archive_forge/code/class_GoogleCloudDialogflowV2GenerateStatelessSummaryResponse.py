from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2GenerateStatelessSummaryResponse(_messages.Message):
    """The response message for Conversations.GenerateStatelessSummary.

  Fields:
    contextSize: Number of messages prior to and including
      last_conversation_message used to compile the suggestion. It may be
      smaller than the GenerateStatelessSummaryRequest.context_size field in
      the request if there weren't that many messages in the conversation.
    latestMessage: The name of the latest conversation message used as context
      for compiling suggestion. The format is specific to the user and the
      names of the messages provided.
    summary: Generated summary.
  """
    contextSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    latestMessage = _messages.StringField(2)
    summary = _messages.MessageField('GoogleCloudDialogflowV2GenerateStatelessSummaryResponseSummary', 3)