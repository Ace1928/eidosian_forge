from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StateNotificationConfig(_messages.Message):
    """The configuration for notification of new states received from the
  device.

  Fields:
    pubsubTopicName: A Cloud Pub/Sub topic name. For example,
      `projects/myProject/topics/deviceEvents`.
  """
    pubsubTopicName = _messages.StringField(1)