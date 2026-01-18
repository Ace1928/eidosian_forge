from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemberRelation(_messages.Message):
    """Message representing a transitive membership of a group.

  Enums:
    RelationTypeValueValuesEnum: The relation between the group and the
      transitive member.

  Fields:
    member: Resource name for this member.
    preferredMemberKey: Entity key has an id and a namespace. In case of
      discussion forums, the id will be an email address without a namespace.
    relationType: The relation between the group and the transitive member.
    roles: The membership role details (i.e name of role and expiry time).
  """

    class RelationTypeValueValuesEnum(_messages.Enum):
        """The relation between the group and the transitive member.

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
    member = _messages.StringField(1)
    preferredMemberKey = _messages.MessageField('EntityKey', 2, repeated=True)
    relationType = _messages.EnumField('RelationTypeValueValuesEnum', 3)
    roles = _messages.MessageField('TransitiveMembershipRole', 4, repeated=True)