from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GroupRelation(_messages.Message):
    """Message representing a transitive group of a user or a group.

  Enums:
    RelationTypeValueValuesEnum: The relation between the member and the
      transitive group.

  Messages:
    LabelsValue: Labels for Group resource.

  Fields:
    displayName: Display name for this group.
    group: Resource name for this group.
    groupKey: Entity key has an id and a namespace. In case of discussion
      forums, the id will be an email address without a namespace.
    labels: Labels for Group resource.
    relationType: The relation between the member and the transitive group.
    roles: Membership roles of the member for the group.
  """

    class RelationTypeValueValuesEnum(_messages.Enum):
        """The relation between the member and the transitive group.

    Values:
      RELATION_TYPE_UNSPECIFIED: The relation type is undefined or
        undetermined.
      DIRECT: The two entities have only a direct membership with each other.
      INDIRECT: The two entities have only an indirect membership with each
        other.
      DIRECT_AND_INDIRECT: The two entities have both a direct and an indirect
        membership with each other.
    """
        RELATION_TYPE_UNSPECIFIED = 0
        DIRECT = 1
        INDIRECT = 2
        DIRECT_AND_INDIRECT = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels for Group resource.

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
    displayName = _messages.StringField(1)
    group = _messages.StringField(2)
    groupKey = _messages.MessageField('EntityKey', 3)
    labels = _messages.MessageField('LabelsValue', 4)
    relationType = _messages.EnumField('RelationTypeValueValuesEnum', 5)
    roles = _messages.MessageField('TransitiveMembershipRole', 6, repeated=True)