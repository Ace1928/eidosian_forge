from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV1NotificationMessage(_messages.Message):
    """Cloud SCC's Notification

  Fields:
    finding: If it's a Finding based notification config, this field will be
      populated.
    notificationConfigName: Name of the notification config that generated
      current notification.
    resource: The Cloud resource tied to this notification's Finding.
  """
    finding = _messages.MessageField('Finding', 1)
    notificationConfigName = _messages.StringField(2)
    resource = _messages.MessageField('GoogleCloudSecuritycenterV1Resource', 3)