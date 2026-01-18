from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2FixedSizeBucketingConfig(_messages.Message):
    """Buckets values based on fixed size ranges. The Bucketing transformation
  can provide all of this functionality, but requires more configuration. This
  message is provided as a convenience to the user for simple bucketing
  strategies. The transformed value will be a hyphenated string of
  {lower_bound}-{upper_bound}. For example, if lower_bound = 10 and
  upper_bound = 20, all values that are within this bucket will be replaced
  with "10-20". This can be used on data of type: double, long. If the bound
  Value type differs from the type of data being transformed, we will first
  attempt converting the type of the data to be transformed to match the type
  of the bound before comparing. See https://cloud.google.com/sensitive-data-
  protection/docs/concepts-bucketing to learn more.

  Fields:
    bucketSize: Required. Size of each bucket (except for minimum and maximum
      buckets). So if `lower_bound` = 10, `upper_bound` = 89, and
      `bucket_size` = 10, then the following buckets would be used: -10,
      10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-89, 89+. Precision
      up to 2 decimals works.
    lowerBound: Required. Lower bound value of buckets. All values less than
      `lower_bound` are grouped together into a single bucket; for example if
      `lower_bound` = 10, then all values less than 10 are replaced with the
      value "-10".
    upperBound: Required. Upper bound value of buckets. All values greater
      than upper_bound are grouped together into a single bucket; for example
      if `upper_bound` = 89, then all values greater than 89 are replaced with
      the value "89+".
  """
    bucketSize = _messages.FloatField(1)
    lowerBound = _messages.MessageField('GooglePrivacyDlpV2Value', 2)
    upperBound = _messages.MessageField('GooglePrivacyDlpV2Value', 3)