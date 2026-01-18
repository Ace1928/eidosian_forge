from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2DialogflowAssistAnswer(_messages.Message):
    """Represents a Dialogflow assist answer.

  Fields:
    answerRecord: The name of answer record, in the format of
      "projects//locations//answerRecords/"
    intentSuggestion: An intent suggestion generated from conversation.
    queryResult: Result from v2 agent.
  """
    answerRecord = _messages.StringField(1)
    intentSuggestion = _messages.MessageField('GoogleCloudDialogflowV2IntentSuggestion', 2)
    queryResult = _messages.MessageField('GoogleCloudDialogflowV2QueryResult', 3)