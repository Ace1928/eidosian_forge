from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildDefinition(_messages.Message):
    """A BuildDefinition object.

  Messages:
    ExternalParametersValue: A ExternalParametersValue object.
    InternalParametersValue: A InternalParametersValue object.

  Fields:
    buildType: A string attribute.
    externalParameters: A ExternalParametersValue attribute.
    internalParameters: A InternalParametersValue attribute.
    resolvedDependencies: A ResourceDescriptor attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExternalParametersValue(_messages.Message):
        """A ExternalParametersValue object.

    Messages:
      AdditionalProperty: An additional property for a ExternalParametersValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExternalParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InternalParametersValue(_messages.Message):
        """A InternalParametersValue object.

    Messages:
      AdditionalProperty: An additional property for a InternalParametersValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InternalParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    buildType = _messages.StringField(1)
    externalParameters = _messages.MessageField('ExternalParametersValue', 2)
    internalParameters = _messages.MessageField('InternalParametersValue', 3)
    resolvedDependencies = _messages.MessageField('ResourceDescriptor', 4, repeated=True)