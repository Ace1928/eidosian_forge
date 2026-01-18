from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipRelation(_messages.Message):
    """Message containing membership relation.

  Messages:
    LabelsValue: One or more label entries that apply to the Group. Currently
      supported labels contain a key with an empty value.

  Fields:
    description: An extended description to help users determine the purpose
      of a `Group`.
    displayName: The display name of the `Group`.
    group: The [resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      `Group`. Shall be of the form `groups/{group_id}`.
    groupKey: The `EntityKey` of the `Group`.
    labels: One or more label entries that apply to the Group. Currently
      supported labels contain a key with an empty value.
    membership: The [resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      `Membership`. Shall be of the form
      `groups/{group_id}/memberships/{membership_id}`.
    roles: The `MembershipRole`s that apply to the `Membership`.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """One or more label entries that apply to the Group. Currently supported
    labels contain a key with an empty value.

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
    group = _messages.StringField(3)
    groupKey = _messages.MessageField('EntityKey', 4)
    labels = _messages.MessageField('LabelsValue', 5)
    membership = _messages.StringField(6)
    roles = _messages.MessageField('MembershipRole', 7, repeated=True)