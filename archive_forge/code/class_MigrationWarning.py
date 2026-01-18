from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MigrationWarning(_messages.Message):
    """Represents migration resource warning information that can be used with
  google.rpc.Status message. MigrationWarning is used to present the user with
  warning information in migration operations.

  Enums:
    CodeValueValuesEnum: The warning code.

  Fields:
    actionItem: Suggested action for solving the warning.
    code: The warning code.
    helpLinks: URL(s) pointing to additional information on handling the
      current warning.
    warningMessage: The localized warning message.
    warningTime: The time the warning occurred.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """The warning code.

    Values:
      WARNING_CODE_UNSPECIFIED: Default value. This value is not used.
      ADAPTATION_WARNING: A warning originated from OS Adaptation.
    """
        WARNING_CODE_UNSPECIFIED = 0
        ADAPTATION_WARNING = 1
    actionItem = _messages.MessageField('LocalizedMessage', 1)
    code = _messages.EnumField('CodeValueValuesEnum', 2)
    helpLinks = _messages.MessageField('Link', 3, repeated=True)
    warningMessage = _messages.MessageField('LocalizedMessage', 4)
    warningTime = _messages.StringField(5)