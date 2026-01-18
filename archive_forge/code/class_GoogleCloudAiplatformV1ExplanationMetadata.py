from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExplanationMetadata(_messages.Message):
    """Metadata describing the Model's input and output for explanation.

  Messages:
    InputsValue: Required. Map from feature names to feature input metadata.
      Keys are the name of the features. Values are the specification of the
      feature. An empty InputMetadata is valid. It describes a text feature
      which has the name specified as the key in ExplanationMetadata.inputs.
      The baseline of the empty feature is chosen by Vertex AI. For Vertex AI-
      provided Tensorflow images, the key can be any friendly name of the
      feature. Once specified, featureAttributions are keyed by this key (if
      not grouped with another feature). For custom images, the key must match
      with the key in instance.
    OutputsValue: Required. Map from output names to output metadata. For
      Vertex AI-provided Tensorflow images, keys can be any user defined
      string that consists of any UTF-8 characters. For custom images, keys
      are the name of the output field in the prediction to be explained.
      Currently only one key is allowed.

  Fields:
    featureAttributionsSchemaUri: Points to a YAML file stored on Google Cloud
      Storage describing the format of the feature attributions. The schema is
      defined as an OpenAPI 3.0.2 [Schema
      Object](https://github.com/OAI/OpenAPI-
      Specification/blob/main/versions/3.0.2.md#schemaObject). AutoML tabular
      Models always have this field populated by Vertex AI. Note: The URI
      given on output may be different, including the URI scheme, than the one
      given on input. The output URI will point to a location where the user
      only has a read access.
    inputs: Required. Map from feature names to feature input metadata. Keys
      are the name of the features. Values are the specification of the
      feature. An empty InputMetadata is valid. It describes a text feature
      which has the name specified as the key in ExplanationMetadata.inputs.
      The baseline of the empty feature is chosen by Vertex AI. For Vertex AI-
      provided Tensorflow images, the key can be any friendly name of the
      feature. Once specified, featureAttributions are keyed by this key (if
      not grouped with another feature). For custom images, the key must match
      with the key in instance.
    latentSpaceSource: Name of the source to generate embeddings for example
      based explanations.
    outputs: Required. Map from output names to output metadata. For Vertex
      AI-provided Tensorflow images, keys can be any user defined string that
      consists of any UTF-8 characters. For custom images, keys are the name
      of the output field in the prediction to be explained. Currently only
      one key is allowed.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InputsValue(_messages.Message):
        """Required. Map from feature names to feature input metadata. Keys are
    the name of the features. Values are the specification of the feature. An
    empty InputMetadata is valid. It describes a text feature which has the
    name specified as the key in ExplanationMetadata.inputs. The baseline of
    the empty feature is chosen by Vertex AI. For Vertex AI-provided
    Tensorflow images, the key can be any friendly name of the feature. Once
    specified, featureAttributions are keyed by this key (if not grouped with
    another feature). For custom images, the key must match with the key in
    instance.

    Messages:
      AdditionalProperty: An additional property for a InputsValue object.

    Fields:
      additionalProperties: Additional properties of type InputsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InputsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1ExplanationMetadataInputMetadata
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1ExplanationMetadataInputMetadata', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class OutputsValue(_messages.Message):
        """Required. Map from output names to output metadata. For Vertex AI-
    provided Tensorflow images, keys can be any user defined string that
    consists of any UTF-8 characters. For custom images, keys are the name of
    the output field in the prediction to be explained. Currently only one key
    is allowed.

    Messages:
      AdditionalProperty: An additional property for a OutputsValue object.

    Fields:
      additionalProperties: Additional properties of type OutputsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a OutputsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1ExplanationMetadataOutputMetadata
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1ExplanationMetadataOutputMetadata', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    featureAttributionsSchemaUri = _messages.StringField(1)
    inputs = _messages.MessageField('InputsValue', 2)
    latentSpaceSource = _messages.StringField(3)
    outputs = _messages.MessageField('OutputsValue', 4)