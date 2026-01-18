from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1DocumentPageImageQualityScores(_messages.Message):
    """Image quality scores for the page image.

  Fields:
    detectedDefects: A list of detected defects.
    qualityScore: The overall quality score. Range `[0, 1]` where `1` is
      perfect quality.
  """
    detectedDefects = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentPageImageQualityScoresDetectedDefect', 1, repeated=True)
    qualityScore = _messages.FloatField(2, variant=_messages.Variant.FLOAT)