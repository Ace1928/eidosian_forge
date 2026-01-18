from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictInstanceImageSegmentationPredictionInstance(_messages.Message):
    """Prediction input format for Image Segmentation.

  Fields:
    content: The image bytes to make the predictions on.
    mimeType: The MIME type of the content of the image. Only the images in
      below listed MIME types are supported. - image/jpeg - image/png
  """
    content = _messages.StringField(1)
    mimeType = _messages.StringField(2)