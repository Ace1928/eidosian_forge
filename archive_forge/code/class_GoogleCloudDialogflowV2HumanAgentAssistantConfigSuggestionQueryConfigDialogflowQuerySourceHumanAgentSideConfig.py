from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionQueryConfigDialogflowQuerySourceHumanAgentSideConfig(_messages.Message):
    """The configuration used for human agent side Dialogflow assist
  suggestion.

  Fields:
    agent: Optional. The name of a dialogflow virtual agent used for intent
      detection and suggestion triggered by human agent. Format:
      `projects//locations//agent`.
  """
    agent = _messages.StringField(1)