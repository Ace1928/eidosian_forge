from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableProgress(_messages.Message):
    """Progress info for copying a table's data to the new cluster.

  Enums:
    StateValueValuesEnum:

  Fields:
    estimatedCopiedBytes: Estimate of the number of bytes copied so far for
      this table. This will eventually reach 'estimated_size_bytes' unless the
      table copy is CANCELLED.
    estimatedSizeBytes: Estimate of the size of the table to be copied.
    state: A StateValueValuesEnum attribute.
  """

    class StateValueValuesEnum(_messages.Enum):
        """StateValueValuesEnum enum type.

    Values:
      STATE_UNSPECIFIED: <no description>
      PENDING: The table has not yet begun copying to the new cluster.
      COPYING: The table is actively being copied to the new cluster.
      COMPLETED: The table has been fully copied to the new cluster.
      CANCELLED: The table was deleted before it finished copying to the new
        cluster. Note that tables deleted after completion will stay marked as
        COMPLETED, not CANCELLED.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        COPYING = 2
        COMPLETED = 3
        CANCELLED = 4
    estimatedCopiedBytes = _messages.IntegerField(1)
    estimatedSizeBytes = _messages.IntegerField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)