from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationProfilesPatchRequest(_messages.Message):
    """A DialogflowProjectsConversationProfilesPatchRequest object.

  Fields:
    googleCloudDialogflowV2ConversationProfile: A
      GoogleCloudDialogflowV2ConversationProfile resource to be passed as the
      request body.
    name: The unique identifier of this conversation profile. Format:
      `projects//locations//conversationProfiles/`.
    updateMask: Required. The mask to control which fields to update.
  """
    googleCloudDialogflowV2ConversationProfile = _messages.MessageField('GoogleCloudDialogflowV2ConversationProfile', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)