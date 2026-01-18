from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserManagedStatus(_messages.Message):
    """The replication status of a SecretVersion using user-managed
  replication. Only populated if the parent Secret has a user-managed
  replication policy.

  Fields:
    replicas: Output only. The list of replica statuses for the SecretVersion.
  """
    replicas = _messages.MessageField('ReplicaStatus', 1, repeated=True)