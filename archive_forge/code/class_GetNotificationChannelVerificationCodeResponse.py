from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetNotificationChannelVerificationCodeResponse(_messages.Message):
    """The GetNotificationChannelVerificationCode request.

  Fields:
    code: The verification code, which may be used to verify other channels
      that have an equivalent identity (i.e. other channels of the same type
      with the same fingerprint such as other email channels with the same
      email address or other sms channels with the same number).
    expireTime: The expiration time associated with the code that was
      returned. If an expiration was provided in the request, this is the
      minimum of the requested expiration in the request and the max permitted
      expiration.
  """
    code = _messages.StringField(1)
    expireTime = _messages.StringField(2)