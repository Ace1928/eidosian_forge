from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EgressPolicyEvalStatesValueListEntryValuesEnum(_messages.Enum):
    """EgressPolicyEvalStatesValueListEntryValuesEnum enum type.

    Values:
      EGRESS_POLICY_EVAL_STATE_UNSPECIFIED: Not used
      EGRESS_POLICY_EVAL_STATE_IN_SAME_SERVICE_PERIMETER: The resources are in
        the same regular service perimeter
      EGRESS_POLICY_EVAL_STATE_GRANTED_OVER_BRIDGE: The resources are in the
        same bridge service perimeter
      EGRESS_POLICY_EVAL_STATE_GRANTED_BY_POLICY: The request is granted by
        the egress policy
      EGRESS_POLICY_EVAL_STATE_DENIED_BY_POLICY: The request is denied by the
        egress policy
      EGRESS_POLICY_EVAL_STATE_NOT_APPLICABLE: The egress policy is applicable
        for the request
    """
    EGRESS_POLICY_EVAL_STATE_UNSPECIFIED = 0
    EGRESS_POLICY_EVAL_STATE_IN_SAME_SERVICE_PERIMETER = 1
    EGRESS_POLICY_EVAL_STATE_GRANTED_OVER_BRIDGE = 2
    EGRESS_POLICY_EVAL_STATE_GRANTED_BY_POLICY = 3
    EGRESS_POLICY_EVAL_STATE_DENIED_BY_POLICY = 4
    EGRESS_POLICY_EVAL_STATE_NOT_APPLICABLE = 5