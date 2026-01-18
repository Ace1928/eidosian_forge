from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EvalStateValueValuesEnum(_messages.Enum):
    """Details about the evaluation state of the vpc accessible service
    policy

    Values:
      EVAL_STATE_UNSPECIFIED: Not used
      NOT_APPLICABLE: Vpc accessible service evaluation is not applicable
      GRANTED: Vpc accessible service policy grants the request
      DENIED: Vpc accessible service policy denies the request
      INTERNAL: It is an internal traffic
    """
    EVAL_STATE_UNSPECIFIED = 0
    NOT_APPLICABLE = 1
    GRANTED = 2
    DENIED = 3
    INTERNAL = 4