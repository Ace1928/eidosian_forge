from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamBinding(_messages.Message):
    """Represents a particular IAM binding, which captures a member's role
  addition, removal, or state.

  Enums:
    ActionValueValuesEnum: The action that was performed on a Binding.

  Fields:
    action: The action that was performed on a Binding.
    member: A single identity requesting access for a Cloud Platform resource,
      for example, "foo@google.com".
    role: Role that is assigned to "members". For example, "roles/viewer",
      "roles/editor", or "roles/owner".
  """

    class ActionValueValuesEnum(_messages.Enum):
        """The action that was performed on a Binding.

    Values:
      ACTION_UNSPECIFIED: Unspecified.
      ADD: Addition of a Binding.
      REMOVE: Removal of a Binding.
    """
        ACTION_UNSPECIFIED = 0
        ADD = 1
        REMOVE = 2
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    member = _messages.StringField(2)
    role = _messages.StringField(3)