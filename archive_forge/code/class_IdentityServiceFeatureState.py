from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityServiceFeatureState(_messages.Message):
    """State for Anthos Identity Service

  Enums:
    StateValueValuesEnum: Deployment state on this member

  Fields:
    failureReason: The reason of the failure.
    installedVersion: Installed AIS version. This is the AIS version installed
      on this member. The values makes sense iff state is OK.
    memberConfig: Membership config state on this member
    state: Deployment state on this member
  """

    class StateValueValuesEnum(_messages.Enum):
        """Deployment state on this member

    Values:
      DEPLOYMENT_STATE_UNSPECIFIED: Unspecified state
      OK: deployment succeeds
      ERROR: Failure with error.
    """
        DEPLOYMENT_STATE_UNSPECIFIED = 0
        OK = 1
        ERROR = 2
    failureReason = _messages.StringField(1)
    installedVersion = _messages.StringField(2)
    memberConfig = _messages.MessageField('MemberConfig', 3)
    state = _messages.EnumField('StateValueValuesEnum', 4)