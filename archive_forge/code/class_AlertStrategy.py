from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlertStrategy(_messages.Message):
    """Control over how the notification channels in notification_channels are
  notified when this alert fires.

  Fields:
    autoClose: If an alert policy that was active has no data for this long,
      any open incidents will close
    notificationChannelStrategy: Control how notifications will be sent out,
      on a per-channel basis.
    notificationRateLimit: Required for alert policies with a LogMatch
      condition.This limit is not implemented for alert policies that are not
      log-based.
  """
    autoClose = _messages.StringField(1)
    notificationChannelStrategy = _messages.MessageField('NotificationChannelStrategy', 2, repeated=True)
    notificationRateLimit = _messages.MessageField('NotificationRateLimit', 3)