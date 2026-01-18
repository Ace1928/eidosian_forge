from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsContaineranalysisV1alpha1SlsaProvenanceZeroTwoSlsaInvocation(_messages.Message):
    """Identifies the event that kicked off the build.

  Messages:
    EnvironmentValue: Any other builder-controlled inputs necessary for
      correctly evaluating the build.
    ParametersValue: Collection of all external inputs that influenced the
      build on top of invocation.configSource.

  Fields:
    configSource: Describes where the config file that kicked off the build
      came from.
    environment: Any other builder-controlled inputs necessary for correctly
      evaluating the build.
    parameters: Collection of all external inputs that influenced the build on
      top of invocation.configSource.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EnvironmentValue(_messages.Message):
        """Any other builder-controlled inputs necessary for correctly evaluating
    the build.

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
        """Collection of all external inputs that influenced the build on top of
    invocation.configSource.

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
    configSource = _messages.MessageField('GoogleDevtoolsContaineranalysisV1alpha1SlsaProvenanceZeroTwoSlsaConfigSource', 1)
    environment = _messages.MessageField('EnvironmentValue', 2)
    parameters = _messages.MessageField('ParametersValue', 3)