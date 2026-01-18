from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RuntimeConfig(_messages.Message):
    """Runtime configuration to run the extension.

  Messages:
    DefaultParamsValue: Optional. Default parameters that will be set for all
      the execution of this extension. If specified, the parameter values can
      be overridden by values in [[ExecuteExtensionRequest.operation_params]]
      at request time. The struct should be in a form of map with param name
      as the key and actual param value as the value. E.g. If this operation
      requires a param "name" to be set to "abc". you can set this to
      something like {"name": "abc"}.

  Fields:
    codeInterpreterRuntimeConfig: Code execution runtime configurations for
      code interpreter extension.
    defaultParams: Optional. Default parameters that will be set for all the
      execution of this extension. If specified, the parameter values can be
      overridden by values in [[ExecuteExtensionRequest.operation_params]] at
      request time. The struct should be in a form of map with param name as
      the key and actual param value as the value. E.g. If this operation
      requires a param "name" to be set to "abc". you can set this to
      something like {"name": "abc"}.
    vertexAiSearchRuntimeConfig: Runtime configuration for Vertext AI Search
      extension.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DefaultParamsValue(_messages.Message):
        """Optional. Default parameters that will be set for all the execution of
    this extension. If specified, the parameter values can be overridden by
    values in [[ExecuteExtensionRequest.operation_params]] at request time.
    The struct should be in a form of map with param name as the key and
    actual param value as the value. E.g. If this operation requires a param
    "name" to be set to "abc". you can set this to something like {"name":
    "abc"}.

    Messages:
      AdditionalProperty: An additional property for a DefaultParamsValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DefaultParamsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    codeInterpreterRuntimeConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1RuntimeConfigCodeInterpreterRuntimeConfig', 1)
    defaultParams = _messages.MessageField('DefaultParamsValue', 2)
    vertexAiSearchRuntimeConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1RuntimeConfigVertexAISearchRuntimeConfig', 3)