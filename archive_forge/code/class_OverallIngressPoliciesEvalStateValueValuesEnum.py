from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OverallIngressPoliciesEvalStateValueValuesEnum(_messages.Enum):
    """Overall evaluation state of the ingress policies

    Values:
      OVERALL_INGRESS_POLICIES_EVAL_STATE_UNSPECIFIED: Not used
      OVERALL_INGRESS_POLICIES_EVAL_STATE_GRANTED: The request is granted by
        the ingress policies
      OVERALL_INGRESS_POLICIES_EVAL_STATE_DENIED: The request is denied by the
        ingress policies
      OVERALL_INGRESS_POLICIES_EVAL_STATE_NOT_APPLICABLE: The ingress policies
        are applicable for the request
      OVERALL_INGRESS_POLICIES_EVAL_STATE_SKIPPED: The request skips the
        ingress policies check
    """
    OVERALL_INGRESS_POLICIES_EVAL_STATE_UNSPECIFIED = 0
    OVERALL_INGRESS_POLICIES_EVAL_STATE_GRANTED = 1
    OVERALL_INGRESS_POLICIES_EVAL_STATE_DENIED = 2
    OVERALL_INGRESS_POLICIES_EVAL_STATE_NOT_APPLICABLE = 3
    OVERALL_INGRESS_POLICIES_EVAL_STATE_SKIPPED = 4