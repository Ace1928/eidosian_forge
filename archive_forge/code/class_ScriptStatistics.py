from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScriptStatistics(_messages.Message):
    """Job statistics specific to the child job of a script.

  Enums:
    EvaluationKindValueValuesEnum: Whether this child job was a statement or
      expression.

  Fields:
    evaluationKind: Whether this child job was a statement or expression.
    stackFrames: Stack trace showing the line/column/procedure name of each
      frame on the stack at the point where the current evaluation happened.
      The leaf frame is first, the primary script is last. Never empty.
  """

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
    evaluationKind = _messages.EnumField('EvaluationKindValueValuesEnum', 1)
    stackFrames = _messages.MessageField('ScriptStackFrame', 2, repeated=True)