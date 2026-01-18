from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsNotificationChannelsVerifyRequest(_messages.Message):
    """A MonitoringProjectsNotificationChannelsVerifyRequest object.

  Fields:
    name: Required. The notification channel to verify.
    verifyNotificationChannelRequest: A VerifyNotificationChannelRequest
      resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    verifyNotificationChannelRequest = _messages.MessageField('VerifyNotificationChannelRequest', 2)