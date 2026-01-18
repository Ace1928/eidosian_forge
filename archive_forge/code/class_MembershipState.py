from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipState(_messages.Message):
    """MembershipState describes the state of a Membership resource.

  Enums:
    CodeValueValuesEnum: Output only. The current state of the Membership
      resource.

  Fields:
    code: Output only. The current state of the Membership resource.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Output only. The current state of the Membership resource.

    Values:
      CODE_UNSPECIFIED: The code is not set.
      CREATING: The cluster is being registered.
      READY: The cluster is registered.
      DELETING: The cluster is being unregistered.
      UPDATING: The Membership is being updated.
      SERVICE_UPDATING: The Membership is being updated by the Hub Service.
    """
        CODE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        DELETING = 3
        UPDATING = 4
        SERVICE_UPDATING = 5
    code = _messages.EnumField('CodeValueValuesEnum', 1)