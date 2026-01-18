from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LockInfo(_messages.Message):
    """Details about the lock which locked the deployment.

  Fields:
    createTime: Time that the lock was taken.
    info: Extra information to store with the lock, provided by the caller.
    lockId: Unique ID for the lock to be overridden with generation ID in the
      backend.
    operation: Terraform operation, provided by the caller.
    version: Terraform version
    who: user@hostname when available
  """
    createTime = _messages.StringField(1)
    info = _messages.StringField(2)
    lockId = _messages.IntegerField(3)
    operation = _messages.StringField(4)
    version = _messages.StringField(5)
    who = _messages.StringField(6)