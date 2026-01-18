from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageSuggestions(_messages.Message):
    """The collection of suggestions.

  Fields:
    suggestions: Required. The list of suggested replies.
  """
    suggestions = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageSuggestion', 1, repeated=True)