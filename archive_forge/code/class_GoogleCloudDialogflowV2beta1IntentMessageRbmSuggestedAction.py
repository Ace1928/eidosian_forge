from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageRbmSuggestedAction(_messages.Message):
    """Rich Business Messaging (RBM) suggested client-side action that the user
  can choose from the card.

  Fields:
    dial: Suggested client side action: Dial a phone number
    openUrl: Suggested client side action: Open a URI on device
    postbackData: Opaque payload that the Dialogflow receives in a user event
      when the user taps the suggested action. This data will be also
      forwarded to webhook to allow performing custom business logic.
    shareLocation: Suggested client side action: Share user location
    text: Text to display alongside the action.
  """
    dial = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageRbmSuggestedActionRbmSuggestedActionDial', 1)
    openUrl = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageRbmSuggestedActionRbmSuggestedActionOpenUri', 2)
    postbackData = _messages.StringField(3)
    shareLocation = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageRbmSuggestedActionRbmSuggestedActionShareLocation', 4)
    text = _messages.StringField(5)