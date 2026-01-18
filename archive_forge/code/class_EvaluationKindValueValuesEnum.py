from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EvaluationKindValueValuesEnum(_messages.Enum):
    """Whether this child job was a statement or expression.

    Values:
      EVALUATION_KIND_UNSPECIFIED: Default value.
      STATEMENT: The statement appears directly in the script.
      EXPRESSION: The statement evaluates an expression that appears in the
        script.
    """
    EVALUATION_KIND_UNSPECIFIED = 0
    STATEMENT = 1
    EXPRESSION = 2