from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateInstancePartitionRequest(_messages.Message):
    """The request for CreateInstancePartition.

  Fields:
    instancePartition: Required. The instance partition to create. The
      instance_partition.name may be omitted, but if specified must be
      `/instancePartitions/`.
    instancePartitionId: Required. The ID of the instance partition to create.
      Valid identifiers are of the form `a-z*[a-z0-9]` and must be between 2
      and 64 characters in length.
  """
    instancePartition = _messages.MessageField('InstancePartition', 1)
    instancePartitionId = _messages.StringField(2)