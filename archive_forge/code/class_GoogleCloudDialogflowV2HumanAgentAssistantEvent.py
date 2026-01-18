from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentAssistantEvent(_messages.Message):
    """Represents a notification sent to Cloud Pub/Sub subscribers for human
  agent assistant events in a specific conversation.

  Fields:
    conversation: The conversation this notification refers to. Format:
      `projects//conversations/`.
    participant: The participant that the suggestion is compiled for. Format:
      `projects//conversations//participants/`. It will not be set in legacy
      workflow.
    suggestionResults: The suggestion results payload that this notification
      refers to.
  """
    conversation = _messages.StringField(1)
    participant = _messages.StringField(2)
    suggestionResults = _messages.MessageField('GoogleCloudDialogflowV2SuggestionResult', 3, repeated=True)