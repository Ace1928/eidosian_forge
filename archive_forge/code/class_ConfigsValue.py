from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ConfigsValue(_messages.Message):
    """Mapping configuration for this SliceSpec. The key is the name of the
    feature. By default, the key will be prefixed by "instance" as a
    dictionary prefix for Vertex Batch Predictions output format.

    Messages:
      AdditionalProperty: An additional property for a ConfigsValue object.

    Fields:
      additionalProperties: Additional properties of type ConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A
          GoogleCloudAiplatformV1ModelEvaluationSliceSliceSliceSpecSliceConfig
          attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudAiplatformV1ModelEvaluationSliceSliceSliceSpecSliceConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)