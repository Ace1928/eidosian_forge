from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserCredential(_messages.Message):
    """Represents a personal access token that authorized the Connection, and
  associated metadata.

  Fields:
    userTokenSecretVersion: Required. A SecretManager resource containing the
      user token that authorizes the Cloud Build connection. Format:
      `projects/*/secrets/*/versions/*`.
    username: Output only. The username associated to this token.
  """
    userTokenSecretVersion = _messages.StringField(1)
    username = _messages.StringField(2)