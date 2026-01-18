from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WhenExpression(_messages.Message):
    """Conditions that need to be true for the task to run.

  Enums:
    ExpressionOperatorValueValuesEnum: Operator that represents an Input's
      relationship to the values

  Fields:
    expressionOperator: Operator that represents an Input's relationship to
      the values
    input: Input is the string for guard checking which can be a static input
      or an output from a parent Task.
    values: Values is an array of strings, which is compared against the
      input, for guard checking.
  """

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
    expressionOperator = _messages.EnumField('ExpressionOperatorValueValuesEnum', 1)
    input = _messages.StringField(2)
    values = _messages.StringField(3, repeated=True)