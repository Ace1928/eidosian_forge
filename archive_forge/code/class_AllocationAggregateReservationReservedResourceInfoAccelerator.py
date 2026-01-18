from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocationAggregateReservationReservedResourceInfoAccelerator(_messages.Message):
    """A AllocationAggregateReservationReservedResourceInfoAccelerator object.

  Fields:
    acceleratorCount: Number of accelerators of specified type.
    acceleratorType: Full or partial URL to accelerator type. e.g.
      "projects/{PROJECT}/zones/{ZONE}/acceleratorTypes/ct4l"
  """
    acceleratorCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    acceleratorType = _messages.StringField(2)