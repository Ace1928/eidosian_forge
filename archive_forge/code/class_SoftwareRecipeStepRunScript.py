from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SoftwareRecipeStepRunScript(_messages.Message):
    """Runs a script through an interpreter.

  Enums:
    InterpreterValueValuesEnum: The script interpreter to use to run the
      script. If no interpreter is specified the script is executed directly,
      which likely only succeed for scripts with [shebang
      lines](https://en.wikipedia.org/wiki/Shebang_\\(Unix\\)).

  Fields:
    allowedExitCodes: Return codes that indicate that the software installed
      or updated successfully. Behaviour defaults to [0]
    interpreter: The script interpreter to use to run the script. If no
      interpreter is specified the script is executed directly, which likely
      only succeed for scripts with [shebang
      lines](https://en.wikipedia.org/wiki/Shebang_\\(Unix\\)).
    script: Required. The shell script to be executed.
  """

    class InterpreterValueValuesEnum(_messages.Enum):
        """The script interpreter to use to run the script. If no interpreter is
    specified the script is executed directly, which likely only succeed for
    scripts with [shebang
    lines](https://en.wikipedia.org/wiki/Shebang_\\(Unix\\)).

    Values:
      INTERPRETER_UNSPECIFIED: Default value for ScriptType.
      SHELL: Indicates that the script is run with `/bin/sh` on Linux and
        `cmd` on windows.
      POWERSHELL: Indicates that the script is run with powershell.
    """
        INTERPRETER_UNSPECIFIED = 0
        SHELL = 1
        POWERSHELL = 2
    allowedExitCodes = _messages.IntegerField(1, repeated=True, variant=_messages.Variant.INT32)
    interpreter = _messages.EnumField('InterpreterValueValuesEnum', 2)
    script = _messages.StringField(3)