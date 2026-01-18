from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1AssetSecurityStatus(_messages.Message):
    """Security policy status of the asset. Data security policy, i.e.,
  readers, writers & owners, should be specified in the lake/zone/asset IAM
  policy.

  Enums:
    StateValueValuesEnum: The current state of the security policy applied to
      the attached resource.

  Fields:
    message: Additional information about the current state.
    state: The current state of the security policy applied to the attached
      resource.
    updateTime: Last update time of the status.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The current state of the security policy applied to the attached
    resource.

    Values:
      STATE_UNSPECIFIED: State unspecified.
      READY: Security policy has been successfully applied to the attached
        resource.
      APPLYING: Security policy is in the process of being applied to the
        attached resource.
      ERROR: Security policy could not be applied to the attached resource due
        to errors.
    """
        STATE_UNSPECIFIED = 0
        READY = 1
        APPLYING = 2
        ERROR = 3
    message = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
    updateTime = _messages.StringField(3)