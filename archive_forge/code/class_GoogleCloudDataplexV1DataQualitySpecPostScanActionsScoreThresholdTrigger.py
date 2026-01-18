from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualitySpecPostScanActionsScoreThresholdTrigger(_messages.Message):
    """This trigger is triggered when the DQ score in the job result is less
  than a specified input score.

  Fields:
    scoreThreshold: Optional. The score range is in 0,100.
  """
    scoreThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)