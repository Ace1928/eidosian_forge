from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IngressSourceEvalStatesValueListEntryValuesEnum(_messages.Enum):
    """IngressSourceEvalStatesValueListEntryValuesEnum enum type.

    Values:
      INGRESS_SOURCE_EVAL_STATE_UNSPECIFIED: Not used
      INGRESS_SOURCE_EVAL_STATE_MATCH: The request matches the ingress source
      INGRESS_SOURCE_EVAL_STATE_NOT_MATCH: The request doesn't match the
        ingress source
    """
    INGRESS_SOURCE_EVAL_STATE_UNSPECIFIED = 0
    INGRESS_SOURCE_EVAL_STATE_MATCH = 1
    INGRESS_SOURCE_EVAL_STATE_NOT_MATCH = 2