from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageImage(_messages.Message):
    """The image response message.

  Fields:
    accessibilityText: A text description of the image to be used for
      accessibility, e.g., screen readers. Required if image_uri is set for
      CarouselSelect.
    imageUri: Optional. The public URI to an image file.
  """
    accessibilityText = _messages.StringField(1)
    imageUri = _messages.StringField(2)