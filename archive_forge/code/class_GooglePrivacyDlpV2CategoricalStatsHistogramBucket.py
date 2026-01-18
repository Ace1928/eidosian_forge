from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CategoricalStatsHistogramBucket(_messages.Message):
    """Histogram of value frequencies in the column.

  Fields:
    bucketSize: Total number of values in this bucket.
    bucketValueCount: Total number of distinct values in this bucket.
    bucketValues: Sample of value frequencies in this bucket. The total number
      of values returned per bucket is capped at 20.
    valueFrequencyLowerBound: Lower bound on the value frequency of the values
      in this bucket.
    valueFrequencyUpperBound: Upper bound on the value frequency of the values
      in this bucket.
  """
    bucketSize = _messages.IntegerField(1)
    bucketValueCount = _messages.IntegerField(2)
    bucketValues = _messages.MessageField('GooglePrivacyDlpV2ValueFrequency', 3, repeated=True)
    valueFrequencyLowerBound = _messages.IntegerField(4)
    valueFrequencyUpperBound = _messages.IntegerField(5)