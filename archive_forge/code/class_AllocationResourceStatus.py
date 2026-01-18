from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocationResourceStatus(_messages.Message):
    """[Output Only] Contains output only fields.

  Fields:
    specificSkuAllocation: Allocation Properties of this reservation.
  """
    specificSkuAllocation = _messages.MessageField('AllocationResourceStatusSpecificSKUAllocation', 1)