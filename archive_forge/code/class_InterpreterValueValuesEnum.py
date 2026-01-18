from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
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