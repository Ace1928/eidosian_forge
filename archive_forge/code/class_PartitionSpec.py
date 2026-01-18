from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartitionSpec(_messages.Message):
    """Specifications of BigQuery partitioned table as export destination.

  Enums:
    PartitionKeyValueValuesEnum: The partition key for BigQuery partitioned
      table.

  Fields:
    partitionKey: The partition key for BigQuery partitioned table.
  """

    class PartitionKeyValueValuesEnum(_messages.Enum):
        """The partition key for BigQuery partitioned table.

    Values:
      PARTITION_KEY_UNSPECIFIED: Unspecified partition key. If used, it means
        using non-partitioned table.
      READ_TIME: The time when the snapshot is taken. If specified as
        partition key, the result table(s) is partitoned by the additional
        timestamp column, readTime. If [read_time] in ExportAssetsRequest is
        specified, the readTime column's value will be the same as it.
        Otherwise, its value will be the current time that is used to take the
        snapshot.
      REQUEST_TIME: The time when the request is received and started to be
        processed. If specified as partition key, the result table(s) is
        partitoned by the requestTime column, an additional timestamp column
        representing when the request was received.
    """
        PARTITION_KEY_UNSPECIFIED = 0
        READ_TIME = 1
        REQUEST_TIME = 2
    partitionKey = _messages.EnumField('PartitionKeyValueValuesEnum', 1)