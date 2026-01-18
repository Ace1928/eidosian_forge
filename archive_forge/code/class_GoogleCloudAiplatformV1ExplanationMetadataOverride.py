from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExplanationMetadataOverride(_messages.Message):
    """The ExplanationMetadata entries that can be overridden at online
  explanation time.

  Messages:
    InputsValue: Required. Overrides the input metadata of the features. The
      key is the name of the feature to be overridden. The keys specified here
      must exist in the input metadata to be overridden. If a feature is not
      specified here, the corresponding feature's input metadata is not
      overridden.

  Fields:
    inputs: Required. Overrides the input metadata of the features. The key is
      the name of the feature to be overridden. The keys specified here must
      exist in the input metadata to be overridden. If a feature is not
      specified here, the corresponding feature's input metadata is not
      overridden.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InputsValue(_messages.Message):
        """Required. Overrides the input metadata of the features. The key is the
    name of the feature to be overridden. The keys specified here must exist
    in the input metadata to be overridden. If a feature is not specified
    here, the corresponding feature's input metadata is not overridden.

    Messages:
      AdditionalProperty: An additional property for a InputsValue object.

    Fields:
      additionalProperties: Additional properties of type InputsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InputsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1ExplanationMetadataOverrideInputMetada
          taOverride attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1ExplanationMetadataOverrideInputMetadataOverride', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    inputs = _messages.MessageField('InputsValue', 1)