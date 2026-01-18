from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrafeasV1SlsaProvenanceZeroTwoSlsaInvocation(_messages.Message):
    """Identifies the event that kicked off the build.

  Messages:
    EnvironmentValue: A EnvironmentValue object.
    ParametersValue: A ParametersValue object.

  Fields:
    configSource: A GrafeasV1SlsaProvenanceZeroTwoSlsaConfigSource attribute.
    environment: A EnvironmentValue attribute.
    parameters: A ParametersValue attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EnvironmentValue(_messages.Message):
        """A EnvironmentValue object.

    Messages:
      AdditionalProperty: An additional property for a EnvironmentValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EnvironmentValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """A ParametersValue object.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    configSource = _messages.MessageField('GrafeasV1SlsaProvenanceZeroTwoSlsaConfigSource', 1)
    environment = _messages.MessageField('EnvironmentValue', 2)
    parameters = _messages.MessageField('ParametersValue', 3)