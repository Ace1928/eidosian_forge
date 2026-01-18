from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2LDiversityHistogramBucket(_messages.Message):
    """Histogram of l-diversity equivalence class sensitive value frequencies.

  Fields:
    bucketSize: Total number of equivalence classes in this bucket.
    bucketValueCount: Total number of distinct equivalence classes in this
      bucket.
    bucketValues: Sample of equivalence classes in this bucket. The total
      number of classes returned per bucket is capped at 20.
    sensitiveValueFrequencyLowerBound: Lower bound on the sensitive value
      frequencies of the equivalence classes in this bucket.
    sensitiveValueFrequencyUpperBound: Upper bound on the sensitive value
      frequencies of the equivalence classes in this bucket.
  """
    bucketSize = _messages.IntegerField(1)
    bucketValueCount = _messages.IntegerField(2)
    bucketValues = _messages.MessageField('GooglePrivacyDlpV2LDiversityEquivalenceClass', 3, repeated=True)
    sensitiveValueFrequencyLowerBound = _messages.IntegerField(4)
    sensitiveValueFrequencyUpperBound = _messages.IntegerField(5)