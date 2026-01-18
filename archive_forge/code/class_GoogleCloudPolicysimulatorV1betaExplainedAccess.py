from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1betaExplainedAccess(_messages.Message):
    """Details about how a set of policies, listed in ExplainedPolicy, resulted
  in a certain AccessState when replaying an access tuple.

  Enums:
    AccessStateValueValuesEnum: Whether the principal in the access tuple has
      permission to access the resource in the access tuple under the given
      policies.

  Fields:
    accessState: Whether the principal in the access tuple has permission to
      access the resource in the access tuple under the given policies.
    errors: If the AccessState is `UNKNOWN`, this field contains a list of
      errors explaining why the result is `UNKNOWN`. If the `AccessState` is
      `GRANTED` or `NOT_GRANTED`, this field is omitted.
    policies: If the AccessState is `UNKNOWN`, this field contains the
      policies that led to that result. If the `AccessState` is `GRANTED` or
      `NOT_GRANTED`, this field is omitted.
  """

    class AccessStateValueValuesEnum(_messages.Enum):
        """Whether the principal in the access tuple has permission to access the
    resource in the access tuple under the given policies.

    Values:
      ACCESS_STATE_UNSPECIFIED: Default value. This value is unused.
      GRANTED: The principal has the permission.
      NOT_GRANTED: The principal does not have the permission.
      UNKNOWN_CONDITIONAL: The principal has the permission only if a
        condition expression evaluates to `true`.
      UNKNOWN_INFO_DENIED: The user who created the Replay does not have
        access to all of the policies that Policy Simulator needs to evaluate.
    """
        ACCESS_STATE_UNSPECIFIED = 0
        GRANTED = 1
        NOT_GRANTED = 2
        UNKNOWN_CONDITIONAL = 3
        UNKNOWN_INFO_DENIED = 4
    accessState = _messages.EnumField('AccessStateValueValuesEnum', 1)
    errors = _messages.MessageField('GoogleRpcStatus', 2, repeated=True)
    policies = _messages.MessageField('GoogleCloudPolicysimulatorV1betaExplainedPolicy', 3, repeated=True)