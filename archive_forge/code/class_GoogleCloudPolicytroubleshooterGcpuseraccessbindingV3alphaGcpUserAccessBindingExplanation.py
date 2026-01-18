from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterGcpuseraccessbindingV3alphaGcpUserAccessBindingExplanation(_messages.Message):
    """The explanation of the GcpUserAccessBinding. NextTAG: 4

  Enums:
    AccessLevelEvaluationStatesValueListEntryValuesEnum:
    EvalStateValueValuesEnum: Output only. Evaluation state of this
      GcpUserAccessBinding.
    PrincipalStateValueValuesEnum: Output only. Principal evaluation states
      indicating whether the principals match.

  Fields:
    accessLevelEvaluationStates: Output only. Access level evaluation states.
    evalState: Output only. Evaluation state of this GcpUserAccessBinding.
    principalState: Output only. Principal evaluation states indicating
      whether the principals match.
  """

    class AccessLevelEvaluationStatesValueListEntryValuesEnum(_messages.Enum):
        """AccessLevelEvaluationStatesValueListEntryValuesEnum enum type.

    Values:
      ACCESS_LEVEL_EVAL_STATE_UNSPECIFIED: Not used
      ACCESS_LEVEL_EVAL_STATE_SATISFIED: The access level is satisfied
      ACCESS_LEVEL_EVAL_STATE_UNSATISFIED: The access level is unsatisfied
      ACCESS_LEVEL_EVAL_STATE_ERROR: The access level is not satisfied nor
        unsatisfied
      ACCESS_LEVEL_EVAL_STATE_NOT_EXIST: The access level does not exist
      ACCESS_LEVEL_EVAL_STATE_INFO_DENIED: No permission to read access
        levels.
    """
        ACCESS_LEVEL_EVAL_STATE_UNSPECIFIED = 0
        ACCESS_LEVEL_EVAL_STATE_SATISFIED = 1
        ACCESS_LEVEL_EVAL_STATE_UNSATISFIED = 2
        ACCESS_LEVEL_EVAL_STATE_ERROR = 3
        ACCESS_LEVEL_EVAL_STATE_NOT_EXIST = 4
        ACCESS_LEVEL_EVAL_STATE_INFO_DENIED = 5

    class EvalStateValueValuesEnum(_messages.Enum):
        """Output only. Evaluation state of this GcpUserAccessBinding.

    Values:
      EVAL_STATE_UNSPECIFIED: Not used
      EVAL_STATE_GRANTED: The GcpUserAccessBinding grants the request.
      EVAL_STATE_DENIED: The GcpUserAccessBinding denies the request.
      EVAL_STATE_NOT_APPLICABLE: The GcpUserAccessBinding is not applicable
        for the principal.
      EVAL_STATE_UNKNOWN: / No enough information to get a conclusion.
    """
        EVAL_STATE_UNSPECIFIED = 0
        EVAL_STATE_GRANTED = 1
        EVAL_STATE_DENIED = 2
        EVAL_STATE_NOT_APPLICABLE = 3
        EVAL_STATE_UNKNOWN = 4

    class PrincipalStateValueValuesEnum(_messages.Enum):
        """Output only. Principal evaluation states indicating whether the
    principals match.

    Values:
      PRINCIPAL_STATE_UNSPECIFIED: Not used
      PRINCIPAL_STATE_MATCHED: Principal matches the GcpUserAccessBinding
        principal.
      PRINCIPAL_STATE_UNMATCHED: Principal does not match the
        GcpUserAccessBinding principal.
      PRINCIPAL_STATE_NOT_FOUND: GcpUserAccessBinding principal does not
        exist.
      PRINCIPAL_STATE_INFO_DENIED: Principal does not have enough permission
        to read the GcpUserAccessBinding principal.
      PRINCIPAL_STATE_UNSUPPORTED: Denied or target principal is not supported
        to troubleshoot.
    """
        PRINCIPAL_STATE_UNSPECIFIED = 0
        PRINCIPAL_STATE_MATCHED = 1
        PRINCIPAL_STATE_UNMATCHED = 2
        PRINCIPAL_STATE_NOT_FOUND = 3
        PRINCIPAL_STATE_INFO_DENIED = 4
        PRINCIPAL_STATE_UNSUPPORTED = 5
    accessLevelEvaluationStates = _messages.EnumField('AccessLevelEvaluationStatesValueListEntryValuesEnum', 1, repeated=True)
    evalState = _messages.EnumField('EvalStateValueValuesEnum', 2)
    principalState = _messages.EnumField('PrincipalStateValueValuesEnum', 3)