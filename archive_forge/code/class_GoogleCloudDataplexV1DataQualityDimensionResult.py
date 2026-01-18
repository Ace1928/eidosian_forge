from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualityDimensionResult(_messages.Message):
    """DataQualityDimensionResult provides a more detailed, per-dimension view
  of the results.

  Fields:
    dimension: Output only. The dimension config specified in the
      DataQualitySpec, as is.
    passed: Whether the dimension passed or failed.
    score: Output only. The dimension-level data quality score for this data
      scan job if and only if the 'dimension' field is set.The score ranges
      between 0, 100 (up to two decimal points).
  """
    dimension = _messages.MessageField('GoogleCloudDataplexV1DataQualityDimension', 1)
    passed = _messages.BooleanField(2)
    score = _messages.FloatField(3, variant=_messages.Variant.FLOAT)