from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class InputArtifactsValue(_messages.Message):
    """The runtime artifacts of the PipelineJob. The key will be the input
    artifact name and the value would be one of the InputArtifact.

    Messages:
      AdditionalProperty: An additional property for a InputArtifactsValue
        object.

    Fields:
      additionalProperties: Additional properties of type InputArtifactsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a InputArtifactsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1PipelineJobRuntimeConfigInputArtifact
          attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudAiplatformV1PipelineJobRuntimeConfigInputArtifact', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)