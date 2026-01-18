from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2KAnonymityHistogramBucket(_messages.Message):
    """Histogram of k-anonymity equivalence classes.

  Fields:
    bucketSize: Total number of equivalence classes in this bucket.
    bucketValueCount: Total number of distinct equivalence classes in this
      bucket.
    bucketValues: Sample of equivalence classes in this bucket. The total
      number of classes returned per bucket is capped at 20.
    equivalenceClassSizeLowerBound: Lower bound on the size of the equivalence
      classes in this bucket.
    equivalenceClassSizeUpperBound: Upper bound on the size of the equivalence
      classes in this bucket.
  """
    bucketSize = _messages.IntegerField(1)
    bucketValueCount = _messages.IntegerField(2)
    bucketValues = _messages.MessageField('GooglePrivacyDlpV2KAnonymityEquivalenceClass', 3, repeated=True)
    equivalenceClassSizeLowerBound = _messages.IntegerField(4)
    equivalenceClassSizeUpperBound = _messages.IntegerField(5)