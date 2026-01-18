from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessKeyCredentials(_messages.Message):
    """Message describing AWS Credentials using access key id and secret.

  Fields:
    accessKeyId: AWS access key ID.
    secretAccessKey: Input only. AWS secret access key.
    sessionToken: Input only. AWS session token. Used only when AWS security
      token service (STS) is responsible for creating the temporary
      credentials.
  """
    accessKeyId = _messages.StringField(1)
    secretAccessKey = _messages.StringField(2)
    sessionToken = _messages.StringField(3)