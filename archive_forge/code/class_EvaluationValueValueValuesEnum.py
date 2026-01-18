from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EvaluationValueValueValuesEnum(_messages.Enum):
    """The evaluation result.

    Values:
      EVALUATION_VALUE_UNSPECIFIED: Reserved for future use.
      TRUE: The evaluation result is `true`.
      FALSE: The evaluation result is `false`.
      CONDITIONAL: The evaluation result is `conditional` when the condition
        expression contains variables that are either missing input values or
        have not been supported by Policy Analyzer yet.
    """
    EVALUATION_VALUE_UNSPECIFIED = 0
    TRUE = 1
    FALSE = 2
    CONDITIONAL = 3