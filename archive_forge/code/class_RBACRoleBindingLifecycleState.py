from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RBACRoleBindingLifecycleState(_messages.Message):
    """RBACRoleBindingLifecycleState describes the state of a RbacRoleBinding
  resource.

  Enums:
    CodeValueValuesEnum: Output only. The current state of the rbacrolebinding
      resource.

  Fields:
    code: Output only. The current state of the rbacrolebinding resource.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Output only. The current state of the rbacrolebinding resource.

    Values:
      CODE_UNSPECIFIED: The code is not set.
      CREATING: The rbacrolebinding is being created.
      READY: The rbacrolebinding active.
      DELETING: The rbacrolebinding is being deleted.
      UPDATING: The rbacrolebinding is being updated.
    """
        CODE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        DELETING = 3
        UPDATING = 4
    code = _messages.EnumField('CodeValueValuesEnum', 1)