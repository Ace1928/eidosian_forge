from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageBrowseCarouselCard(_messages.Message):
    """Browse Carousel Card for Actions on Google.
  https://developers.google.com/actions/assistant/responses#browsing_carousel

  Enums:
    ImageDisplayOptionsValueValuesEnum: Optional. Settings for displaying the
      image. Applies to every image in items.

  Fields:
    imageDisplayOptions: Optional. Settings for displaying the image. Applies
      to every image in items.
    items: Required. List of items in the Browse Carousel Card. Minimum of two
      items, maximum of ten.
  """

    class ImageDisplayOptionsValueValuesEnum(_messages.Enum):
        """Optional. Settings for displaying the image. Applies to every image in
    items.

    Values:
      IMAGE_DISPLAY_OPTIONS_UNSPECIFIED: Fill the gaps between the image and
        the image container with gray bars.
      GRAY: Fill the gaps between the image and the image container with gray
        bars.
      WHITE: Fill the gaps between the image and the image container with
        white bars.
      CROPPED: Image is scaled such that the image width and height match or
        exceed the container dimensions. This may crop the top and bottom of
        the image if the scaled image height is greater than the container
        height, or crop the left and right of the image if the scaled image
        width is greater than the container width. This is similar to "Zoom
        Mode" on a widescreen TV when playing a 4:3 video.
      BLURRED_BACKGROUND: Pad the gaps between image and image frame with a
        blurred copy of the same image.
    """
        IMAGE_DISPLAY_OPTIONS_UNSPECIFIED = 0
        GRAY = 1
        WHITE = 2
        CROPPED = 3
        BLURRED_BACKGROUND = 4
    imageDisplayOptions = _messages.EnumField('ImageDisplayOptionsValueValuesEnum', 1)
    items = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageBrowseCarouselCardBrowseCarouselCardItem', 2, repeated=True)