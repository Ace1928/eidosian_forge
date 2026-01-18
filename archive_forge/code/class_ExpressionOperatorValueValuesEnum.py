from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExpressionOperatorValueValuesEnum(_messages.Enum):
    """Operator that represents an Input's relationship to the values

    Values:
      EXPRESSION_OPERATOR_UNSPECIFIED: Default enum type; should not be used.
      IN: Input is in values.
      NOT_IN: Input is not in values.
    """
    EXPRESSION_OPERATOR_UNSPECIFIED = 0
    IN = 1
    NOT_IN = 2