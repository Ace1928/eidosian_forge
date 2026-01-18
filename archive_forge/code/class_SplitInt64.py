from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SplitInt64(_messages.Message):
    """A representation of an int64, n, that is immune to precision loss when
  encoded in JSON.

  Fields:
    highBits: The high order bits, including the sign: n >> 32.
    lowBits: The low order bits: n & 0xffffffff.
  """
    highBits = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    lowBits = _messages.IntegerField(2, variant=_messages.Variant.UINT32)