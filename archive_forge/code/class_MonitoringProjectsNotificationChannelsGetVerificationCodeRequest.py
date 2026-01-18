from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsNotificationChannelsGetVerificationCodeRequest(_messages.Message):
    """A MonitoringProjectsNotificationChannelsGetVerificationCodeRequest
  object.

  Fields:
    getNotificationChannelVerificationCodeRequest: A
      GetNotificationChannelVerificationCodeRequest resource to be passed as
      the request body.
    name: Required. The notification channel for which a verification code is
      to be generated and retrieved. This must name a channel that is already
      verified; if the specified channel is not verified, the request will
      fail.
  """
    getNotificationChannelVerificationCodeRequest = _messages.MessageField('GetNotificationChannelVerificationCodeRequest', 1)
    name = _messages.StringField(2, required=True)