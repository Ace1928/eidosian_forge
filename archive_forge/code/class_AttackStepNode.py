from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttackStepNode(_messages.Message):
    """Detailed steps the attack can take between path nodes.

  Enums:
    TypeValueValuesEnum: Attack step type. Can be either AND, OR or DEFENSE

  Messages:
    LabelsValue: Attack step labels for metadata

  Fields:
    description: Attack step description
    displayName: User friendly name of the attack step
    labels: Attack step labels for metadata
    type: Attack step type. Can be either AND, OR or DEFENSE
    uuid: Unique ID for one Node
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Attack step type. Can be either AND, OR or DEFENSE

    Values:
      NODE_TYPE_UNSPECIFIED: Type not specified
      NODE_TYPE_AND: Incoming edge joined with AND
      NODE_TYPE_OR: Incoming edge joined with OR
      NODE_TYPE_DEFENSE: Incoming edge is defense
      NODE_TYPE_ATTACKER: Incoming edge is attacker
    """
        NODE_TYPE_UNSPECIFIED = 0
        NODE_TYPE_AND = 1
        NODE_TYPE_OR = 2
        NODE_TYPE_DEFENSE = 3
        NODE_TYPE_ATTACKER = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Attack step labels for metadata

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    type = _messages.EnumField('TypeValueValuesEnum', 4)
    uuid = _messages.StringField(5)