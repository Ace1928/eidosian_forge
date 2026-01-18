from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowHistogramValue(_messages.Message):
    """Summary statistics for a population of values. HistogramValue contains a
  sequence of buckets and gives a count of values that fall into each bucket.
  Bucket boundares are defined by a formula and bucket widths are either fixed
  or exponentially increasing.

  Fields:
    bucketCounts: Optional. The number of values in each bucket of the
      histogram, as described in `bucket_options`. `bucket_counts` should
      contain N values, where N is the number of buckets specified in
      `bucket_options`. If `bucket_counts` has fewer than N values, the
      remaining values are assumed to be 0.
    bucketOptions: Describes the bucket boundaries used in the histogram.
    count: Number of values recorded in this histogram.
    outlierStats: Statistics on the values recorded in the histogram that fall
      out of the bucket boundaries.
  """
    bucketCounts = _messages.IntegerField(1, repeated=True)
    bucketOptions = _messages.MessageField('BucketOptions', 2)
    count = _messages.IntegerField(3)
    outlierStats = _messages.MessageField('OutlierStats', 4)