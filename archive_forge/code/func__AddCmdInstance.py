from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def _AddCmdInstance(command_name, cmd, command_aliases=None):
    """Add a command from a Cmd instance.

  Args:
    command_name:    name of the command which will be used in argument parsing
    cmd:             Cmd instance to register
    command_aliases: A list of command aliases that the command can be run as.

  Raises:
    AppCommandsError: is command is already registered OR cmd is not a subclass
                      of Cmd
    AppCommandsError: if name is already registered OR name is not a string OR
                      name is too short OR name does not start with a letter OR
                      name contains any non alphanumeric characters besides
                      '_', '-', or ':'.
  """
    global _cmd_list
    global _cmd_alias_list
    if not issubclass(cmd.__class__, Cmd):
        raise AppCommandsError('Command must be an instance of commands.Cmd')
    for name in [command_name] + (command_aliases or []):
        _CheckCmdName(name)
        _cmd_alias_list[name] = command_name
    _cmd_list[command_name] = cmd