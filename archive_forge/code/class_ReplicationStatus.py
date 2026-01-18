from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReplicationStatus(_messages.Message):
    """The replication status of a SecretVersion.

  Fields:
    automatic: Describes the replication status of a SecretVersion with
      automatic replication. Only populated if the parent Secret has an
      automatic replication policy.
    userManaged: Describes the replication status of a SecretVersion with
      user-managed replication. Only populated if the parent Secret has a
      user-managed replication policy.
  """
    automatic = _messages.MessageField('AutomaticStatus', 1)
    userManaged = _messages.MessageField('UserManagedStatus', 2)