from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocateLunsRequestLunRange(_messages.Message):
    """A LUN(Logical Unit Number) range.

  Fields:
    quantity: Number of LUNs to create.
    sizeGb: The requested size of each LUN, in GB.
  """
    quantity = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    sizeGb = _messages.IntegerField(2, variant=_messages.Variant.INT32)