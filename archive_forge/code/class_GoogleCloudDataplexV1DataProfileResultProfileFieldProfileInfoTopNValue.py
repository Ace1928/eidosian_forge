from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataProfileResultProfileFieldProfileInfoTopNValue(_messages.Message):
    """Top N non-null values in the scanned data.

  Fields:
    count: Count of the corresponding value in the scanned data.
    ratio: Ratio of the corresponding value in the field against the total
      number of rows in the scanned data.
    value: String value of a top N non-null value.
  """
    count = _messages.IntegerField(1)
    ratio = _messages.FloatField(2)
    value = _messages.StringField(3)