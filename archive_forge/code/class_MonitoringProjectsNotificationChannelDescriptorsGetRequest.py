from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsNotificationChannelDescriptorsGetRequest(_messages.Message):
    """A MonitoringProjectsNotificationChannelDescriptorsGetRequest object.

  Fields:
    name: Required. The channel type for which to execute the request. The
      format is: projects/[PROJECT_ID_OR_NUMBER]/notificationChannelDescriptor
      s/[CHANNEL_TYPE]
  """
    name = _messages.StringField(1, required=True)