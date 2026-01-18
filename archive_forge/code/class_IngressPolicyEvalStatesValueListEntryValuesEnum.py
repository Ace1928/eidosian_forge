from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IngressPolicyEvalStatesValueListEntryValuesEnum(_messages.Enum):
    """IngressPolicyEvalStatesValueListEntryValuesEnum enum type.

    Values:
      INGRESS_POLICY_EVAL_STATE_UNSPECIFIED: Not used
      INGRESS_POLICY_EVAL_STATE_IN_SAME_SERVICE_PERIMETER: The resources are
        in the same regular service perimeter
      INGRESS_POLICY_EVAL_STATE_GRANTED_OVER_BRIDGE: The resources are in the
        same bridge service perimeter
      INGRESS_POLICY_EVAL_STATE_GRANTED_BY_POLICY: The request is granted by
        the ingress policy
      INGRESS_POLICY_EVAL_STATE_DENIED_BY_POLICY: The request is denied by the
        ingress policy
      INGRESS_POLICY_EVAL_STATE_NOT_APPLICABLE: The ingress policy is
        applicable for the request
    """
    INGRESS_POLICY_EVAL_STATE_UNSPECIFIED = 0
    INGRESS_POLICY_EVAL_STATE_IN_SAME_SERVICE_PERIMETER = 1
    INGRESS_POLICY_EVAL_STATE_GRANTED_OVER_BRIDGE = 2
    INGRESS_POLICY_EVAL_STATE_GRANTED_BY_POLICY = 3
    INGRESS_POLICY_EVAL_STATE_DENIED_BY_POLICY = 4
    INGRESS_POLICY_EVAL_STATE_NOT_APPLICABLE = 5