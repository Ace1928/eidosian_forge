from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PropertiesInfo(_messages.Message):
    """Properties of the workload organized by origin.

  Messages:
    AutotuningPropertiesValue: Output only. Properties set by autotuning
      engine.

  Fields:
    autotuningProperties: Output only. Properties set by autotuning engine.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AutotuningPropertiesValue(_messages.Message):
        """Output only. Properties set by autotuning engine.

    Messages:
      AdditionalProperty: An additional property for a
        AutotuningPropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type
        AutotuningPropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AutotuningPropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A ValueInfo attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('ValueInfo', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    autotuningProperties = _messages.MessageField('AutotuningPropertiesValue', 1)