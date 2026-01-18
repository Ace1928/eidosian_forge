from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1App(_messages.Message):
    """A GoogleCloudApigeeV1App object.

  Fields:
    apiProducts: List of API products associated with the app.
    appGroup: Name of the AppGroup
    appId: ID of the app.
    attributes: List of attributes.
    callbackUrl: Callback URL used by OAuth 2.0 authorization servers to
      communicate authorization codes back to apps.
    companyName: Name of the company that owns the app.
    createdAt: Output only. Unix time when the app was created.
    credentials: Output only. Set of credentials for the app. Credentials are
      API key/secret pairs associated with API products.
    developerEmail: Email of the developer.
    developerId: ID of the developer.
    keyExpiresIn: Duration, in milliseconds, of the consumer key that will be
      generated for the app. The default value, -1, indicates an infinite
      validity period. Once set, the expiration can't be updated. json key:
      keyExpiresIn
    lastModifiedAt: Output only. Last modified time as milliseconds since
      epoch.
    name: Name of the app.
    scopes: Scopes to apply to the app. The specified scope names must already
      exist on the API product that you associate with the app.
    status: Status of the credential.
  """
    apiProducts = _messages.MessageField('GoogleCloudApigeeV1ApiProductRef', 1, repeated=True)
    appGroup = _messages.StringField(2)
    appId = _messages.StringField(3)
    attributes = _messages.MessageField('GoogleCloudApigeeV1Attribute', 4, repeated=True)
    callbackUrl = _messages.StringField(5)
    companyName = _messages.StringField(6)
    createdAt = _messages.IntegerField(7)
    credentials = _messages.MessageField('GoogleCloudApigeeV1Credential', 8, repeated=True)
    developerEmail = _messages.StringField(9)
    developerId = _messages.StringField(10)
    keyExpiresIn = _messages.IntegerField(11)
    lastModifiedAt = _messages.IntegerField(12)
    name = _messages.StringField(13)
    scopes = _messages.StringField(14, repeated=True)
    status = _messages.StringField(15)