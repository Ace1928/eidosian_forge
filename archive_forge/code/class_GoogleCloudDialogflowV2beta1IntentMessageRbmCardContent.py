from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageRbmCardContent(_messages.Message):
    """Rich Business Messaging (RBM) Card content

  Fields:
    description: Optional. Description of the card (at most 2000 bytes). At
      least one of the title, description or media must be set.
    media: Optional. However at least one of the title, description or media
      must be set. Media (image, GIF or a video) to include in the card.
    suggestions: Optional. List of suggestions to include in the card.
    title: Optional. Title of the card (at most 200 bytes). At least one of
      the title, description or media must be set.
  """
    description = _messages.StringField(1)
    media = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageRbmCardContentRbmMedia', 2)
    suggestions = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageRbmSuggestion', 3, repeated=True)
    title = _messages.StringField(4)