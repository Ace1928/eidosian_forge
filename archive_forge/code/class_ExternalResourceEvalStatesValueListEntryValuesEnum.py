from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExternalResourceEvalStatesValueListEntryValuesEnum(_messages.Enum):
    """ExternalResourceEvalStatesValueListEntryValuesEnum enum type.

    Values:
      RESOURCE_EVAL_STATE_UNSPECIFIED: Not used
      RESOURCE_EVAL_STATE_MATCH: The request matches the resource
      RESOURCE_EVAL_STATE_NOT_MATCH: The request doesn't match the resource
    """
    RESOURCE_EVAL_STATE_UNSPECIFIED = 0
    RESOURCE_EVAL_STATE_MATCH = 1
    RESOURCE_EVAL_STATE_NOT_MATCH = 2