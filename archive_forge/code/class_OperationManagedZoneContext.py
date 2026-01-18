from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OperationManagedZoneContext(_messages.Message):
    """A OperationManagedZoneContext object.

  Fields:
    newValue: The post-operation ManagedZone resource.
    oldValue: The pre-operation ManagedZone resource.
  """
    newValue = _messages.MessageField('ManagedZone', 1)
    oldValue = _messages.MessageField('ManagedZone', 2)