from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageSuggestion(_messages.Message):
    """The suggestion chip message that the user can tap to quickly post a
  reply to the conversation.

  Fields:
    title: Required. The text shown the in the suggestion chip.
  """
    title = _messages.StringField(1)