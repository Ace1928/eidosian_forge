from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1ExponentialBuckets(_messages.Message):
    """Describing buckets with exponentially growing width.

  Fields:
    growthFactor: The i'th exponential bucket covers the interval [scale *
      growth_factor^(i-1), scale * growth_factor^i) where i ranges from 1 to
      num_finite_buckets inclusive. Must be larger than 1.0.
    numFiniteBuckets: The number of finite buckets. With the underflow and
      overflow buckets, the total number of buckets is `num_finite_buckets` +
      2. See comments on `bucket_options` for details.
    scale: The i'th exponential bucket covers the interval [scale *
      growth_factor^(i-1), scale * growth_factor^i) where i ranges from 1 to
      num_finite_buckets inclusive. Must be > 0.
  """
    growthFactor = _messages.FloatField(1)
    numFiniteBuckets = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    scale = _messages.FloatField(3)