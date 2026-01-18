from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartitionedColumn(_messages.Message):
    """The partitioning column information.

  Fields:
    field: Output only. The name of the partition column.
  """
    field = _messages.StringField(1)