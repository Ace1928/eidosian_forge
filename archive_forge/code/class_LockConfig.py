from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LockConfig(_messages.Message):
    """Determines whether or no a connection is locked. If locked, a reason
  must be specified.

  Fields:
    locked: Indicates whether or not the connection is locked.
    reason: Describes why a connection is locked.
  """
    locked = _messages.BooleanField(1)
    reason = _messages.StringField(2)