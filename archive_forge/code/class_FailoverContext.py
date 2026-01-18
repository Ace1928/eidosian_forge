from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class FailoverContext(_messages.Message):
    """Database instance failover context.

  Fields:
    kind: This is always `sql#failoverContext`.
    settingsVersion: The current settings version of this instance. Request
      will be rejected if this version doesn't match the current settings
      version.
  """
    kind = _messages.StringField(1)
    settingsVersion = _messages.IntegerField(2)