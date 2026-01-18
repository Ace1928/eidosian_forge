from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageBasicCard(_messages.Message):
    """The basic card message. Useful for displaying information.

  Fields:
    buttons: Optional. The collection of card buttons.
    formattedText: Required, unless image is present. The body text of the
      card.
    image: Optional. The image for the card.
    subtitle: Optional. The subtitle of the card.
    title: Optional. The title of the card.
  """
    buttons = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageBasicCardButton', 1, repeated=True)
    formattedText = _messages.StringField(2)
    image = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageImage', 3)
    subtitle = _messages.StringField(4)
    title = _messages.StringField(5)