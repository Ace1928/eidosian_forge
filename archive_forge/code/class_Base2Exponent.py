from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Base2Exponent(_messages.Message):
    """Exponential buckets where the growth factor between buckets is
  `2**(2**-scale)`. e.g. for `scale=1` growth factor is
  `2**(2**(-1))=sqrt(2)`. `n` buckets will have the following boundaries. -
  0th: [0, gf) - i in [1, n-1]: [gf^(i), gf^(i+1))

  Fields:
    numberOfBuckets: Must be greater than 0.
    scale: Must be between -3 and 3. This forces the growth factor of the
      bucket boundaries to be between `2^(1/8)` and `256`.
  """
    numberOfBuckets = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    scale = _messages.IntegerField(2, variant=_messages.Variant.INT32)