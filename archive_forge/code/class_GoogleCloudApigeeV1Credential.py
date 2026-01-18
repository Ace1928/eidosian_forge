from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Credential(_messages.Message):
    """A GoogleCloudApigeeV1Credential object.

  Fields:
    apiProducts: List of API products this credential can be used for.
    attributes: List of attributes associated with this credential.
    consumerKey: Consumer key.
    consumerSecret: Secret key.
    expiresAt: Time the credential will expire in milliseconds since epoch.
    issuedAt: Time the credential was issued in milliseconds since epoch.
    scopes: List of scopes to apply to the app. Specified scopes must already
      exist on the API product that you associate with the app.
    status: Status of the credential. Valid values include `approved` or
      `revoked`.
  """
    apiProducts = _messages.MessageField('GoogleCloudApigeeV1ApiProductRef', 1, repeated=True)
    attributes = _messages.MessageField('GoogleCloudApigeeV1Attribute', 2, repeated=True)
    consumerKey = _messages.StringField(3)
    consumerSecret = _messages.StringField(4)
    expiresAt = _messages.IntegerField(5)
    issuedAt = _messages.IntegerField(6)
    scopes = _messages.StringField(7, repeated=True)
    status = _messages.StringField(8)