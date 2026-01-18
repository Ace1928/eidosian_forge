from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionQueryConfigContextFilterSettings(_messages.Message):
    """Settings that determine how to filter recent conversation context when
  generating suggestions.

  Fields:
    dropHandoffMessages: If set to true, the last message from virtual agent
      (hand off message) and the message before it (trigger message of hand
      off) are dropped.
    dropIvrMessages: If set to true, all messages from ivr stage are dropped.
    dropVirtualAgentMessages: If set to true, all messages from virtual agent
      are dropped.
  """
    dropHandoffMessages = _messages.BooleanField(1)
    dropIvrMessages = _messages.BooleanField(2)
    dropVirtualAgentMessages = _messages.BooleanField(3)