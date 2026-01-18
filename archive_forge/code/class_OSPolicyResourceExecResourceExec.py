from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourceExecResourceExec(_messages.Message):
    """A file or script to execute.

  Enums:
    InterpreterValueValuesEnum: Required. The script interpreter to use.

  Fields:
    args: Optional arguments to pass to the source during execution.
    file: A remote or local file.
    interpreter: Required. The script interpreter to use.
    outputFilePath: Only recorded for enforce Exec. Path to an output file
      (that is created by this Exec) whose content will be recorded in
      OSPolicyResourceCompliance after a successful run. Absence or failure to
      read this file will result in this ExecResource being non-compliant.
      Output file size is limited to 100K bytes.
    script: An inline script. The size of the script is limited to 32KiB.
  """

    class InterpreterValueValuesEnum(_messages.Enum):
        """Required. The script interpreter to use.

    Values:
      INTERPRETER_UNSPECIFIED: Invalid value, the request will return
        validation error.
      NONE: If an interpreter is not specified, the source is executed
        directly. This execution, without an interpreter, only succeeds for
        executables and scripts that have shebang lines.
      SHELL: Indicates that the script runs with `/bin/sh` on Linux and
        `cmd.exe` on Windows.
      POWERSHELL: Indicates that the script runs with PowerShell.
    """
        INTERPRETER_UNSPECIFIED = 0
        NONE = 1
        SHELL = 2
        POWERSHELL = 3
    args = _messages.StringField(1, repeated=True)
    file = _messages.MessageField('OSPolicyResourceFile', 2)
    interpreter = _messages.EnumField('InterpreterValueValuesEnum', 3)
    outputFilePath = _messages.StringField(4)
    script = _messages.StringField(5)