from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RBACRoleBindingActuationMembershipState(_messages.Message):
    """**RBAC RoleBinding Actuation**: An empty state left as an example
  membership-specific Feature state.

  Enums:
    LifecycleStateValueValuesEnum:

  Fields:
    lifecycleState: A LifecycleStateValueValuesEnum attribute.
    stateDetails: A string attribute.
  """

    class LifecycleStateValueValuesEnum(_messages.Enum):
        """LifecycleStateValueValuesEnum enum type.

    Values:
      LIFECYCLE_STATE_UNSPECIFIED: The lifecycle state is unspecified.
      ACTIVE: <no description>
      ERROR: <no description>
    """
        LIFECYCLE_STATE_UNSPECIFIED = 0
        ACTIVE = 1
        ERROR = 2
    lifecycleState = _messages.EnumField('LifecycleStateValueValuesEnum', 1)
    stateDetails = _messages.StringField(2)