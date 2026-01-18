from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListInstancePartitionsResponse(_messages.Message):
    """The response for ListInstancePartitions.

  Fields:
    instancePartitions: The list of requested instancePartitions.
    nextPageToken: `next_page_token` can be sent in a subsequent
      ListInstancePartitions call to fetch more of the matching instance
      partitions.
    unreachable: The list of unreachable instance partitions. It includes the
      names of instance partitions whose metadata could not be retrieved
      within instance_partition_deadline.
  """
    instancePartitions = _messages.MessageField('InstancePartition', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)