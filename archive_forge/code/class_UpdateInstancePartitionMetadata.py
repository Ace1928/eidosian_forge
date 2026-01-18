from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateInstancePartitionMetadata(_messages.Message):
    """Metadata type for the operation returned by UpdateInstancePartition.

  Fields:
    cancelTime: The time at which this operation was cancelled. If set, this
      operation is in the process of undoing itself (which is guaranteed to
      succeed) and cannot be cancelled again.
    endTime: The time at which this operation failed or was completed
      successfully.
    instancePartition: The desired end state of the update.
    startTime: The time at which UpdateInstancePartition request was received.
  """
    cancelTime = _messages.StringField(1)
    endTime = _messages.StringField(2)
    instancePartition = _messages.MessageField('InstancePartition', 3)
    startTime = _messages.StringField(4)