from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionImageRAIScores(_messages.Message):
    """RAI scores for generated image returned.

  Fields:
    agileWatermarkDetectionScore: Agile watermark score for image.
  """
    agileWatermarkDetectionScore = _messages.FloatField(1)