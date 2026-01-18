from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OauthClientCredential(_messages.Message):
    """Represents an oauth client credential. Used to authenticate an oauth
  client while accessing Google Cloud resources on behalf of a user by using
  OAuth2 Protocol.

  Fields:
    clientSecret: Output only. The system-generated oauth client secret.
    createTime: Output only. The timestamp when the oauth client credential
      was created
    disabled: Optional. Whether the oauth client credential is disabled. You
      cannot use a disabled oauth client credential for OAuth.
    displayName: Optional. A user-specified display name of the oauth client
      credential Cannot exceed 32 characters.
    name: Immutable. The resource name of the oauth client credential. Format:
      `projects/{project}/locations/{location}/oauthClients/{oauth_client}/cre
      dentials/{credential}`
    updateTime: Output only. The timestamp for the last update of the oauth
      client credential. If no updates have been made, the creation time will
      serve as the designated value.
  """
    clientSecret = _messages.StringField(1)
    createTime = _messages.StringField(2)
    disabled = _messages.BooleanField(3)
    displayName = _messages.StringField(4)
    name = _messages.StringField(5)
    updateTime = _messages.StringField(6)