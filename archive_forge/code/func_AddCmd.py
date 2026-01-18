from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def AddCmd(command_name, cmd_factory, **kargs):
    """Add a command from a Cmd subclass or factory.

  Args:
    command_name:    name of the command which will be used in argument parsing
    cmd_factory:     A callable whose arguments match those of Cmd.__init__ and
                     returns a Cmd. In the simplest case this is just a subclass
                     of Cmd.
    command_aliases: A list of command aliases that the command can be run as.

  Raises:
    AppCommandsError: if calling cmd_factory does not return an instance of Cmd.
  """
    cmd = cmd_factory(command_name, flags.FlagValues(), **kargs)
    if not isinstance(cmd, Cmd):
        raise AppCommandsError('Command must be an instance of commands.Cmd')
    _AddCmdInstance(command_name, cmd, **kargs)