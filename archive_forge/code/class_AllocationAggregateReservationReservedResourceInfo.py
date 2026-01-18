from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocationAggregateReservationReservedResourceInfo(_messages.Message):
    """A AllocationAggregateReservationReservedResourceInfo object.

  Fields:
    accelerator: Properties of accelerator resources in this reservation.
  """
    accelerator = _messages.MessageField('AllocationAggregateReservationReservedResourceInfoAccelerator', 1)