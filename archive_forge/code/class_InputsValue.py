from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class InputsValue(_messages.Message):
    """Output only. The runtime input artifacts of the task.

    Messages:
      AdditionalProperty: An additional property for a InputsValue object.

    Fields:
      additionalProperties: Additional properties of type InputsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a InputsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1PipelineTaskDetailArtifactList
          attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudAiplatformV1PipelineTaskDetailArtifactList', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)