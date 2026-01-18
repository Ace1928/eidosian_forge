from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1AuthConfigApiKeyConfig(_messages.Message):
    """Config for authentication with API key.

  Enums:
    HttpElementLocationValueValuesEnum: Required. The location of the API key.

  Fields:
    apiKeySecret: Required. The name of the SecretManager secret version
      resource storing the API key. Format:
      `projects/{project}/secrets/{secrete}/versions/{version}` - If
      specified, the `secretmanager.versions.access` permission should be
      granted to Vertex AI Extension Service Agent
      (https://cloud.google.com/vertex-ai/docs/general/access-control#service-
      agents) on the specified resource.
    httpElementLocation: Required. The location of the API key.
    name: Required. The parameter name of the API key. E.g. If the API request
      is "https://example.com/act?api_key=", "api_key" would be the parameter
      name.
  """

    class HttpElementLocationValueValuesEnum(_messages.Enum):
        """Required. The location of the API key.

    Values:
      HTTP_IN_UNSPECIFIED: <no description>
      HTTP_IN_QUERY: Element is in the HTTP request query.
      HTTP_IN_HEADER: Element is in the HTTP request header.
      HTTP_IN_PATH: Element is in the HTTP request path.
      HTTP_IN_BODY: Element is in the HTTP request body.
      HTTP_IN_COOKIE: Element is in the HTTP request cookie.
    """
        HTTP_IN_UNSPECIFIED = 0
        HTTP_IN_QUERY = 1
        HTTP_IN_HEADER = 2
        HTTP_IN_PATH = 3
        HTTP_IN_BODY = 4
        HTTP_IN_COOKIE = 5
    apiKeySecret = _messages.StringField(1)
    httpElementLocation = _messages.EnumField('HttpElementLocationValueValuesEnum', 2)
    name = _messages.StringField(3)