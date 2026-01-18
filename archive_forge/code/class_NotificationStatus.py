from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotificationStatus(_messages.Message):
    """Status of notification action

  Fields:
    lastUpdateTime: Output only. Time at which the last pub/sub action
      occurred.
    status: Output only. Status of the last pub/sub action.
  """
    lastUpdateTime = _messages.StringField(1)
    status = _messages.MessageField('Status', 2)