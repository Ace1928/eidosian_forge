from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageRbmSuggestion(_messages.Message):
    """Rich Business Messaging (RBM) suggestion. Suggestions allow user to
  easily select/click a predefined response or perform an action (like opening
  a web uri).

  Fields:
    action: Predefined client side actions that user can choose
    reply: Predefined replies for user to select instead of typing
  """
    action = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageRbmSuggestedAction', 1)
    reply = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageRbmSuggestedReply', 2)