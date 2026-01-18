from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuantityBasedExpiry(_messages.Message):
    """A backup's position in a quantity-based retention queue, of backups with
  the same source cluster and type, with length, retention, specified by the
  backup's retention policy. Once the position is greater than the retention,
  the backup is eligible to be garbage collected. Example: 5 backups from the
  same source cluster and type with a quantity-based retention of 3 and
  denoted by backup_id (position, retention). Safe: backup_5 (1, 3), backup_4,
  (2, 3), backup_3 (3, 3). Awaiting garbage collection: backup_2 (4, 3),
  backup_1 (5, 3)

  Fields:
    retentionCount: Output only. The backup's position among its backups with
      the same source cluster and type, by descending chronological order
      create time(i.e. newest first).
    totalRetentionCount: Output only. The length of the quantity-based queue,
      specified by the backup's retention policy.
  """
    retentionCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    totalRetentionCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)