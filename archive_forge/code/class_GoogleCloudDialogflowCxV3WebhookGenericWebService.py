from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3WebhookGenericWebService(_messages.Message):
    """Represents configuration for a generic web service.

  Enums:
    HttpMethodValueValuesEnum: Optional. HTTP method for the flexible webhook
      calls. Standard webhook always uses POST.
    ServiceAgentAuthValueValuesEnum: Optional. Indicate the auth token type
      generated from the [Diglogflow service
      agent](https://cloud.google.com/iam/docs/service-agents#dialogflow-
      service-agent). The generated token is sent in the Authorization header.
    WebhookTypeValueValuesEnum: Optional. Type of the webhook.

  Messages:
    ParameterMappingValue: Optional. Maps the values extracted from specific
      fields of the flexible webhook response into session parameters. - Key:
      session parameter name - Value: field path in the webhook response
    RequestHeadersValue: The HTTP request headers to send together with
      webhook requests.

  Fields:
    allowedCaCerts: Optional. Specifies a list of allowed custom CA
      certificates (in DER format) for HTTPS verification. This overrides the
      default SSL trust store. If this is empty or unspecified, Dialogflow
      will use Google's default trust store to verify certificates. N.B. Make
      sure the HTTPS server certificates are signed with "subject alt name".
      For instance a certificate can be self-signed using the following
      command, ``` openssl x509 -req -days 200 -in example.com.csr \\ -signkey
      example.com.key \\ -out example.com.crt \\ -extfile <(printf
      "\\nsubjectAltName='DNS:www.example.com'") ```
    httpMethod: Optional. HTTP method for the flexible webhook calls. Standard
      webhook always uses POST.
    oauthConfig: Optional. The OAuth configuration of the webhook. If
      specified, Dialogflow will initiate the OAuth client credential flow to
      exchange an access token from the 3rd party platform and put it in the
      auth header.
    parameterMapping: Optional. Maps the values extracted from specific fields
      of the flexible webhook response into session parameters. - Key: session
      parameter name - Value: field path in the webhook response
    password: The password for HTTP Basic authentication.
    requestBody: Optional. Defines a custom JSON object as request body to
      send to flexible webhook.
    requestHeaders: The HTTP request headers to send together with webhook
      requests.
    serviceAgentAuth: Optional. Indicate the auth token type generated from
      the [Diglogflow service
      agent](https://cloud.google.com/iam/docs/service-agents#dialogflow-
      service-agent). The generated token is sent in the Authorization header.
    uri: Required. The webhook URI for receiving POST requests. It must use
      https protocol.
    username: The user name for HTTP Basic authentication.
    webhookType: Optional. Type of the webhook.
  """

    class HttpMethodValueValuesEnum(_messages.Enum):
        """Optional. HTTP method for the flexible webhook calls. Standard webhook
    always uses POST.

    Values:
      HTTP_METHOD_UNSPECIFIED: HTTP method not specified.
      POST: HTTP POST Method.
      GET: HTTP GET Method.
      HEAD: HTTP HEAD Method.
      PUT: HTTP PUT Method.
      DELETE: HTTP DELETE Method.
      PATCH: HTTP PATCH Method.
      OPTIONS: HTTP OPTIONS Method.
    """
        HTTP_METHOD_UNSPECIFIED = 0
        POST = 1
        GET = 2
        HEAD = 3
        PUT = 4
        DELETE = 5
        PATCH = 6
        OPTIONS = 7

    class ServiceAgentAuthValueValuesEnum(_messages.Enum):
        """Optional. Indicate the auth token type generated from the [Diglogflow
    service agent](https://cloud.google.com/iam/docs/service-
    agents#dialogflow-service-agent). The generated token is sent in the
    Authorization header.

    Values:
      SERVICE_AGENT_AUTH_UNSPECIFIED: Service agent auth type unspecified.
        Default to ID_TOKEN.
      NONE: No token used.
      ID_TOKEN: Use [ID
        token](https://cloud.google.com/docs/authentication/token-types#id)
        generated from service agent. This can be used to access Cloud
        Function and Cloud Run after you grant Invoker role to `service-@gcp-
        sa-dialogflow.iam.gserviceaccount.com`.
      ACCESS_TOKEN: Use [access
        token](https://cloud.google.com/docs/authentication/token-
        types#access) generated from service agent. This can be used to access
        other Google Cloud APIs after you grant required roles to
        `service-@gcp-sa-dialogflow.iam.gserviceaccount.com`.
    """
        SERVICE_AGENT_AUTH_UNSPECIFIED = 0
        NONE = 1
        ID_TOKEN = 2
        ACCESS_TOKEN = 3

    class WebhookTypeValueValuesEnum(_messages.Enum):
        """Optional. Type of the webhook.

    Values:
      WEBHOOK_TYPE_UNSPECIFIED: Default value. This value is unused.
      STANDARD: Represents a standard webhook.
      FLEXIBLE: Represents a flexible webhook.
    """
        WEBHOOK_TYPE_UNSPECIFIED = 0
        STANDARD = 1
        FLEXIBLE = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParameterMappingValue(_messages.Message):
        """Optional. Maps the values extracted from specific fields of the
    flexible webhook response into session parameters. - Key: session
    parameter name - Value: field path in the webhook response

    Messages:
      AdditionalProperty: An additional property for a ParameterMappingValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ParameterMappingValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParameterMappingValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class RequestHeadersValue(_messages.Message):
        """The HTTP request headers to send together with webhook requests.

    Messages:
      AdditionalProperty: An additional property for a RequestHeadersValue
        object.

    Fields:
      additionalProperties: Additional properties of type RequestHeadersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a RequestHeadersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    allowedCaCerts = _messages.BytesField(1, repeated=True)
    httpMethod = _messages.EnumField('HttpMethodValueValuesEnum', 2)
    oauthConfig = _messages.MessageField('GoogleCloudDialogflowCxV3WebhookGenericWebServiceOAuthConfig', 3)
    parameterMapping = _messages.MessageField('ParameterMappingValue', 4)
    password = _messages.StringField(5)
    requestBody = _messages.StringField(6)
    requestHeaders = _messages.MessageField('RequestHeadersValue', 7)
    serviceAgentAuth = _messages.EnumField('ServiceAgentAuthValueValuesEnum', 8)
    uri = _messages.StringField(9)
    username = _messages.StringField(10)
    webhookType = _messages.EnumField('WebhookTypeValueValuesEnum', 11)