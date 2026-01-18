from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OverallEgressPoliciesEvalStateValueValuesEnum(_messages.Enum):
    """Overall evaluation state of the egress policies

    Values:
      OVERALL_EGRESS_POLICIES_EVAL_STATE_UNSPECIFIED: Not used
      OVERALL_EGRESS_POLICIES_EVAL_STATE_GRANTED: The request is granted by
        the egress policies
      OVERALL_EGRESS_POLICIES_EVAL_STATE_DENIED: The request is denied by the
        egress policies
      OVERALL_EGRESS_POLICIES_EVAL_STATE_NOT_APPLICABLE: The egress policies
        are applicable for the request
      OVERALL_EGRESS_POLICIES_EVAL_STATE_SKIPPED: The request skips the egress
        policies check
    """
    OVERALL_EGRESS_POLICIES_EVAL_STATE_UNSPECIFIED = 0
    OVERALL_EGRESS_POLICIES_EVAL_STATE_GRANTED = 1
    OVERALL_EGRESS_POLICIES_EVAL_STATE_DENIED = 2
    OVERALL_EGRESS_POLICIES_EVAL_STATE_NOT_APPLICABLE = 3
    OVERALL_EGRESS_POLICIES_EVAL_STATE_SKIPPED = 4