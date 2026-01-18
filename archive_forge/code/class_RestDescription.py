from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RestDescription(_messages.Message):
    """A RestDescription object.

  Messages:
    AuthValue: Authentication information.
    IconsValue: Links to 16x16 and 32x32 icons representing the API.
    MethodsValue: API-level methods for this API.
    ParametersValue: Common parameters that apply across all apis.
    ResourcesValue: The resources in this API.
    SchemasValue: The schemas for this API.

  Fields:
    auth: Authentication information.
    basePath: [DEPRECATED] The base path for REST requests.
    baseUrl: [DEPRECATED] The base URL for REST requests.
    batchPath: The path for REST batch requests.
    canonicalName: Indicates how the API name should be capitalized and split
      into various parts. Useful for generating pretty class names.
    description: The description of this API.
    discoveryVersion: Indicate the version of the Discovery API used to
      generate this doc.
    documentationLink: A link to human readable documentation for the API.
    etag: The etag for this response.
    features: A list of supported features for this API.
    icons: Links to 16x16 and 32x32 icons representing the API.
    id: The id of this API.
    kind: The kind for this response.
    labels: Labels for the status of this API, such as labs or deprecated.
    methods: API-level methods for this API.
    name: The name of this API.
    parameters: Common parameters that apply across all apis.
    protocol: The protocol described by this document.
    resources: The resources in this API.
    revision: The version of this API.
    rootUrl: The root url under which all API services live.
    schemas: The schemas for this API.
    servicePath: The base path for all REST requests.
    title: The title of this API.
    version: The version of this API.
  """

    class AuthValue(_messages.Message):
        """Authentication information.

    Messages:
      Oauth2Value: OAuth 2.0 authentication information.

    Fields:
      oauth2: OAuth 2.0 authentication information.
    """

        class Oauth2Value(_messages.Message):
            """OAuth 2.0 authentication information.

      Messages:
        ScopesValue: Available OAuth 2.0 scopes.

      Fields:
        scopes: Available OAuth 2.0 scopes.
      """

            @encoding.MapUnrecognizedFields('additionalProperties')
            class ScopesValue(_messages.Message):
                """Available OAuth 2.0 scopes.

        Messages:
          AdditionalProperty: An additional property for a ScopesValue object.

        Fields:
          additionalProperties: The scope value.
        """

                class AdditionalProperty(_messages.Message):
                    """An additional property for a ScopesValue object.

          Messages:
            ValueValue: A ValueValue object.

          Fields:
            key: Name of the additional property.
            value: A ValueValue attribute.
          """

                    class ValueValue(_messages.Message):
                        """A ValueValue object.

            Fields:
              description: Description of scope.
            """
                        description = _messages.StringField(1)
                    key = _messages.StringField(1)
                    value = _messages.MessageField('ValueValue', 2)
                additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
            scopes = _messages.MessageField('ScopesValue', 1)
        oauth2 = _messages.MessageField('Oauth2Value', 1)

    class IconsValue(_messages.Message):
        """Links to 16x16 and 32x32 icons representing the API.

    Fields:
      x16: The url of the 16x16 icon.
      x32: The url of the 32x32 icon.
    """
        x16 = _messages.StringField(1)
        x32 = _messages.StringField(2)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MethodsValue(_messages.Message):
        """API-level methods for this API.

    Messages:
      AdditionalProperty: An additional property for a MethodsValue object.

    Fields:
      additionalProperties: An individual method description.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MethodsValue object.

      Fields:
        key: Name of the additional property.
        value: A RestMethod attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('RestMethod', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """Common parameters that apply across all apis.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Description of a single parameter.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A JsonSchema attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('JsonSchema', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourcesValue(_messages.Message):
        """The resources in this API.

    Messages:
      AdditionalProperty: An additional property for a ResourcesValue object.

    Fields:
      additionalProperties: An individual resource description. Contains
        methods and sub-resources related to this resource.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A RestResource attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('RestResource', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SchemasValue(_messages.Message):
        """The schemas for this API.

    Messages:
      AdditionalProperty: An additional property for a SchemasValue object.

    Fields:
      additionalProperties: An individual schema description.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SchemasValue object.

      Fields:
        key: Name of the additional property.
        value: A JsonSchema attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('JsonSchema', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    auth = _messages.MessageField('AuthValue', 1)
    basePath = _messages.StringField(2)
    baseUrl = _messages.StringField(3)
    batchPath = _messages.StringField(4, default='batch')
    canonicalName = _messages.StringField(5)
    description = _messages.StringField(6)
    discoveryVersion = _messages.StringField(7, default='v1')
    documentationLink = _messages.StringField(8)
    etag = _messages.StringField(9)
    features = _messages.StringField(10, repeated=True)
    icons = _messages.MessageField('IconsValue', 11)
    id = _messages.StringField(12)
    kind = _messages.StringField(13, default='discovery#restDescription')
    labels = _messages.StringField(14, repeated=True)
    methods = _messages.MessageField('MethodsValue', 15)
    name = _messages.StringField(16)
    parameters = _messages.MessageField('ParametersValue', 17)
    protocol = _messages.StringField(18, default='rest')
    resources = _messages.MessageField('ResourcesValue', 19)
    revision = _messages.StringField(20)
    rootUrl = _messages.StringField(21)
    schemas = _messages.MessageField('SchemasValue', 22)
    servicePath = _messages.StringField(23)
    title = _messages.StringField(24)
    version = _messages.StringField(25)