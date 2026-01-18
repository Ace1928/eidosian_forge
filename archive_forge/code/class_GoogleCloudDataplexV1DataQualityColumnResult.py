from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualityColumnResult(_messages.Message):
    """DataQualityColumnResult provides a more detailed, per-column view of the
  results.

  Fields:
    column: Output only. The column specified in the DataQualityRule.
    score: Output only. The column-level data quality score for this data scan
      job if and only if the 'column' field is set.The score ranges between
      between 0, 100 (up to two decimal points).
  """
    column = _messages.StringField(1)
    score = _messages.FloatField(2, variant=_messages.Variant.FLOAT)