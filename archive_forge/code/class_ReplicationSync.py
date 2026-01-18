from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReplicationSync(_messages.Message):
    """ReplicationSync contain information about the last replica sync to the
  cloud.

  Fields:
    lastSyncTime: The most updated snapshot created time in the source that
      finished replication.
  """
    lastSyncTime = _messages.StringField(1)