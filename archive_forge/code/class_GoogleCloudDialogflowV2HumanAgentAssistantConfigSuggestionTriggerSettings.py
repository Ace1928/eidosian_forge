from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionTriggerSettings(_messages.Message):
    """Settings of suggestion trigger.

  Fields:
    noSmalltalk: Do not trigger if last utterance is small talk.
    onlyEndUser: Only trigger suggestion if participant role of last utterance
      is END_USER.
  """
    noSmalltalk = _messages.BooleanField(1)
    onlyEndUser = _messages.BooleanField(2)