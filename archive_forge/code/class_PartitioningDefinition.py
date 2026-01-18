from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartitioningDefinition(_messages.Message):
    """The partitioning information, which includes managed table and external
  table partition information.

  Fields:
    partitionedColumn: Output only. Details about each partitioning column.
      BigQuery native tables only support 1 partitioning column. Other table
      types may support 0, 1 or more partitioning columns.
  """
    partitionedColumn = _messages.MessageField('PartitionedColumn', 1, repeated=True)