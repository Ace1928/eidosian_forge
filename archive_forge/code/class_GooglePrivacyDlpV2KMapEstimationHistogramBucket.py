from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2KMapEstimationHistogramBucket(_messages.Message):
    """A KMapEstimationHistogramBucket message with the following values:
  min_anonymity: 3 max_anonymity: 5 frequency: 42 means that there are 42
  records whose quasi-identifier values correspond to 3, 4 or 5 people in the
  overlying population. An important particular case is when min_anonymity =
  max_anonymity = 1: the frequency field then corresponds to the number of
  uniquely identifiable records.

  Fields:
    bucketSize: Number of records within these anonymity bounds.
    bucketValueCount: Total number of distinct quasi-identifier tuple values
      in this bucket.
    bucketValues: Sample of quasi-identifier tuple values in this bucket. The
      total number of classes returned per bucket is capped at 20.
    maxAnonymity: Always greater than or equal to min_anonymity.
    minAnonymity: Always positive.
  """
    bucketSize = _messages.IntegerField(1)
    bucketValueCount = _messages.IntegerField(2)
    bucketValues = _messages.MessageField('GooglePrivacyDlpV2KMapEstimationQuasiIdValues', 3, repeated=True)
    maxAnonymity = _messages.IntegerField(4)
    minAnonymity = _messages.IntegerField(5)