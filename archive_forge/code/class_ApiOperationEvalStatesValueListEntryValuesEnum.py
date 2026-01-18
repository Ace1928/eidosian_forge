from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApiOperationEvalStatesValueListEntryValuesEnum(_messages.Enum):
    """ApiOperationEvalStatesValueListEntryValuesEnum enum type.

    Values:
      API_OPERATION_EVAL_STATE_UNSPECIFIED: Not used
      API_OPERATION_EVAL_STATE_MATCH: The request matches the api operation
      API_OPERATION_EVAL_STATE_NOT_MATCH: The request doesn't match the api
        operation
    """
    API_OPERATION_EVAL_STATE_UNSPECIFIED = 0
    API_OPERATION_EVAL_STATE_MATCH = 1
    API_OPERATION_EVAL_STATE_NOT_MATCH = 2