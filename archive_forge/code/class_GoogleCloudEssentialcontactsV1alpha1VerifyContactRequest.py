from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudEssentialcontactsV1alpha1VerifyContactRequest(_messages.Message):
    """Request message for the VerifyContact method.

  Fields:
    verificationToken: Token, extracted from link in verification email.
  """
    verificationToken = _messages.StringField(1)