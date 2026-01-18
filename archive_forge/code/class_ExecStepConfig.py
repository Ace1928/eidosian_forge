from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ExecStepConfig(_messages.Message):
    """Common configurations for an ExecStep.

  Enums:
    InterpreterValueValuesEnum: The script interpreter to use to run the
      script. If no interpreter is specified the script will be executed
      directly, which will likely only succeed for scripts with [shebang
      lines] (https://en.wikipedia.org/wiki/Shebang_\\(Unix\\)).

  Fields:
    allowedSuccessCodes: Defaults to [0]. A list of possible return values
      that the execution can return to indicate a success.
    gcsObject: A Google Cloud Storage object containing the executable.
    interpreter: The script interpreter to use to run the script. If no
      interpreter is specified the script will be executed directly, which
      will likely only succeed for scripts with [shebang lines]
      (https://en.wikipedia.org/wiki/Shebang_\\(Unix\\)).
    localPath: An absolute path to the executable on the VM.
  """

    class InterpreterValueValuesEnum(_messages.Enum):
        """The script interpreter to use to run the script. If no interpreter is
    specified the script will be executed directly, which will likely only
    succeed for scripts with [shebang lines]
    (https://en.wikipedia.org/wiki/Shebang_\\(Unix\\)).

    Values:
      INTERPRETER_UNSPECIFIED: If the interpreter is not specified, the value
        defaults to `NONE`.
      NONE: Indicates that the file is run as follows on each operating
        system: + For Linux VMs, the file is ran as an executable and the
        interpreter might be parsed from the [shebang
        line](https://wikipedia.org/wiki/Shebang_(Unix)) of the file. + For
        Windows VM, this value is not supported.
      SHELL: Indicates that the file is run with `/bin/sh` on Linux and `cmd`
        on Windows.
      POWERSHELL: Indicates that the file is run with PowerShell.
    """
        INTERPRETER_UNSPECIFIED = 0
        NONE = 1
        SHELL = 2
        POWERSHELL = 3
    allowedSuccessCodes = _messages.IntegerField(1, repeated=True, variant=_messages.Variant.INT32)
    gcsObject = _messages.MessageField('GcsObject', 2)
    interpreter = _messages.EnumField('InterpreterValueValuesEnum', 3)
    localPath = _messages.StringField(4)