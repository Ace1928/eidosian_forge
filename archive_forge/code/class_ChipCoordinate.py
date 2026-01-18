from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChipCoordinate(_messages.Message):
    """Represents a single chip in a logical traffic matrix.

  Fields:
    sliceCoordinate: Coordinate of slice that chip is in.
    xCoordinate: X coordinate of chip.
    yCoordinate: Y coordinate of chip.
    zCoordinate: Z coordinate of chip.
  """
    sliceCoordinate = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    xCoordinate = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    yCoordinate = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    zCoordinate = _messages.IntegerField(4, variant=_messages.Variant.INT32)