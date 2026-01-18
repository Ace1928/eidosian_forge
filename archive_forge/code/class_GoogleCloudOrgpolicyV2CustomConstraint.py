from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudOrgpolicyV2CustomConstraint(_messages.Message):
    """A custom constraint defined by customers which can *only* be applied to
  the given resource types and organization. By creating a custom constraint,
  customers can apply policies of this custom constraint. *Creating a custom
  constraint itself does NOT apply any policy enforcement*.

  Enums:
    ActionTypeValueValuesEnum: Allow or deny type.
    MethodTypesValueListEntryValuesEnum:

  Fields:
    actionType: Allow or deny type.
    condition: Org policy condition/expression. For example:
      `resource.instanceName.matches("[production|test]_.*_(\\d)+")` or,
      `resource.management.auto_upgrade == true` The max length of the
      condition is 1000 characters.
    description: Detailed information about this custom policy constraint. The
      max length of the description is 2000 characters.
    displayName: One line display name for the UI. The max length of the
      display_name is 200 characters.
    methodTypes: All the operations being applied for this constraint.
    name: Immutable. Name of the constraint. This is unique within the
      organization. Format of the name should be * `organizations/{organizatio
      n_id}/customConstraints/{custom_constraint_id}` Example:
      `organizations/123/customConstraints/custom.createOnlyE2TypeVms` The max
      length is 70 characters and the minimum length is 1. Note that the
      prefix `organizations/{organization_id}/customConstraints/` is not
      counted.
    resourceTypes: Immutable. The resource instance type on which this policy
      applies. Format will be of the form : `/` Example: *
      `compute.googleapis.com/Instance`.
    updateTime: Output only. The last time this custom constraint was updated.
      This represents the last time that the `CreateCustomConstraint` or
      `UpdateCustomConstraint` RPC was called
  """

    class ActionTypeValueValuesEnum(_messages.Enum):
        """Allow or deny type.

    Values:
      ACTION_TYPE_UNSPECIFIED: Unspecified. Results in an error.
      ALLOW: Allowed action type.
      DENY: Deny action type.
    """
        ACTION_TYPE_UNSPECIFIED = 0
        ALLOW = 1
        DENY = 2

    class MethodTypesValueListEntryValuesEnum(_messages.Enum):
        """MethodTypesValueListEntryValuesEnum enum type.

    Values:
      METHOD_TYPE_UNSPECIFIED: Unspecified. Results in an error.
      CREATE: Constraint applied when creating the resource.
      UPDATE: Constraint applied when updating the resource.
      DELETE: Constraint applied when deleting the resource. Not supported
        yet.
    """
        METHOD_TYPE_UNSPECIFIED = 0
        CREATE = 1
        UPDATE = 2
        DELETE = 3
    actionType = _messages.EnumField('ActionTypeValueValuesEnum', 1)
    condition = _messages.StringField(2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    methodTypes = _messages.EnumField('MethodTypesValueListEntryValuesEnum', 5, repeated=True)
    name = _messages.StringField(6)
    resourceTypes = _messages.StringField(7, repeated=True)
    updateTime = _messages.StringField(8)