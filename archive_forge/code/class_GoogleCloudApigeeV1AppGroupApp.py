from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AppGroupApp(_messages.Message):
    """Response for [GetAppGroupApp].[AppGroupApps.GetAppGroupApp],
  [CreateAppGroupAppRequest].[AppGroupApp.CreateAppGroupAppRequest] and
  [DeleteAppGroupApp].[AppGroupApp.DeleteAppGroupApp]

  Fields:
    apiProducts: List of API products associated with the AppGroup app.
    appGroup: Immutable. Name of the parent AppGroup whose resource name
      format is of syntax (organizations/*/appgroups/*).
    appId: Immutable. ID of the AppGroup app.
    attributes: List of attributes for the AppGroup app.
    callbackUrl: Callback URL used by OAuth 2.0 authorization servers to
      communicate authorization codes back to AppGroup apps.
    createdAt: Output only. Time the AppGroup app was created in milliseconds
      since epoch.
    credentials: Output only. Set of credentials for the AppGroup app
      consisting of the consumer key/secret pairs associated with the API
      products.
    keyExpiresIn: Immutable. Expiration time, in seconds, for the consumer key
      that is generated for the AppGroup app. If not set or left to the
      default value of `-1`, the API key never expires. The expiration time
      can't be updated after it is set.
    lastModifiedAt: Output only. Time the AppGroup app was modified in
      milliseconds since epoch.
    name: Immutable. Name of the AppGroup app whose resource name format is of
      syntax (organizations/*/appgroups/*/apps/*).
    scopes: Scopes to apply to the AppGroup app. The specified scopes must
      already exist for the API product that you associate with the AppGroup
      app.
    status: Status of the App. Valid values include `approved` or `revoked`.
  """
    apiProducts = _messages.StringField(1, repeated=True)
    appGroup = _messages.StringField(2)
    appId = _messages.StringField(3)
    attributes = _messages.MessageField('GoogleCloudApigeeV1Attribute', 4, repeated=True)
    callbackUrl = _messages.StringField(5)
    createdAt = _messages.IntegerField(6)
    credentials = _messages.MessageField('GoogleCloudApigeeV1Credential', 7, repeated=True)
    keyExpiresIn = _messages.IntegerField(8)
    lastModifiedAt = _messages.IntegerField(9)
    name = _messages.StringField(10)
    scopes = _messages.StringField(11, repeated=True)
    status = _messages.StringField(12)