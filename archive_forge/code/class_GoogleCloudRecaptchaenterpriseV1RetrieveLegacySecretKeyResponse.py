from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1RetrieveLegacySecretKeyResponse(_messages.Message):
    """Secret key is used only in legacy reCAPTCHA. It must be used in a 3rd
  party integration with legacy reCAPTCHA.

  Fields:
    legacySecretKey: The secret key (also known as shared secret) authorizes
      communication between your application backend and the reCAPTCHA
      Enterprise server to create an assessment. The secret key needs to be
      kept safe for security purposes.
  """
    legacySecretKey = _messages.StringField(1)