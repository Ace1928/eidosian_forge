from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScriptOptions(_messages.Message):
    """Options related to script execution.

  Enums:
    KeyResultStatementValueValuesEnum: Determines which statement in the
      script represents the "key result", used to populate the schema and
      query results of the script job. Default is LAST.

  Fields:
    keyResultStatement: Determines which statement in the script represents
      the "key result", used to populate the schema and query results of the
      script job. Default is LAST.
    statementByteBudget: Limit on the number of bytes billed per statement.
      Exceeding this budget results in an error.
    statementTimeoutMs: Timeout period for each statement in a script.
  """

    class KeyResultStatementValueValuesEnum(_messages.Enum):
        """Determines which statement in the script represents the "key result",
    used to populate the schema and query results of the script job. Default
    is LAST.

    Values:
      KEY_RESULT_STATEMENT_KIND_UNSPECIFIED: Default value.
      LAST: The last result determines the key result.
      FIRST_SELECT: The first SELECT statement determines the key result.
    """
        KEY_RESULT_STATEMENT_KIND_UNSPECIFIED = 0
        LAST = 1
        FIRST_SELECT = 2
    keyResultStatement = _messages.EnumField('KeyResultStatementValueValuesEnum', 1)
    statementByteBudget = _messages.IntegerField(2)
    statementTimeoutMs = _messages.IntegerField(3)