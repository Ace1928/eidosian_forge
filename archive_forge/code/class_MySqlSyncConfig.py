from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class MySqlSyncConfig(_messages.Message):
    """MySQL-specific external server sync settings.

  Fields:
    initialSyncFlags: Flags to use for the initial dump.
  """
    initialSyncFlags = _messages.MessageField('SyncFlags', 1, repeated=True)