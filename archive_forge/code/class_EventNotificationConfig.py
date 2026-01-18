from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventNotificationConfig(_messages.Message):
    """The configuration for forwarding telemetry events.

  Fields:
    pubsubTopicName: A Cloud Pub/Sub topic name. For example,
      `projects/myProject/topics/deviceEvents`.
    subfolderMatches: If the subfolder name matches this string exactly, this
      configuration will be used. The string must not include the leading '/'
      character. If empty, all strings are matched. This field is used only
      for telemetry events; subfolders are not supported for state changes.
  """
    pubsubTopicName = _messages.StringField(1)
    subfolderMatches = _messages.StringField(2)