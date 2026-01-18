from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictPredictionImageSegmentationPredictionResult(_messages.Message):
    """Prediction output format for Image Segmentation.

  Fields:
    categoryMask: A PNG image where each pixel in the mask represents the
      category in which the pixel in the original image was predicted to
      belong to. The size of this image will be the same as the original
      image. The mapping between the AnntoationSpec and the color can be found
      in model's metadata. The model will choose the most likely category and
      if none of the categories reach the confidence threshold, the pixel will
      be marked as background.
    confidenceMask: A one channel image which is encoded as an 8bit lossless
      PNG. The size of the image will be the same as the original image. For a
      specific pixel, darker color means less confidence in correctness of the
      cateogry in the categoryMask for the corresponding pixel. Black means no
      confidence and white means complete confidence.
  """
    categoryMask = _messages.StringField(1)
    confidenceMask = _messages.StringField(2)