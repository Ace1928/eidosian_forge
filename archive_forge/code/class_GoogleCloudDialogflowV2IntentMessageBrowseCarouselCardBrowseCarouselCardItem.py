from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageBrowseCarouselCardBrowseCarouselCardItem(_messages.Message):
    """Browsing carousel tile

  Fields:
    description: Optional. Description of the carousel item. Maximum of four
      lines of text.
    footer: Optional. Text that appears at the bottom of the Browse Carousel
      Card. Maximum of one line of text.
    image: Optional. Hero image for the carousel item.
    openUriAction: Required. Action to present to the user.
    title: Required. Title of the carousel item. Maximum of two lines of text.
  """
    description = _messages.StringField(1)
    footer = _messages.StringField(2)
    image = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageImage', 3)
    openUriAction = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageBrowseCarouselCardBrowseCarouselCardItemOpenUrlAction', 4)
    title = _messages.StringField(5)