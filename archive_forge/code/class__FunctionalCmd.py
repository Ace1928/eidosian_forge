from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
class _FunctionalCmd(Cmd):
    """Class to wrap functions as CMD instances.

  Args:
    cmd_func:   command function
  """

    def __init__(self, name, flag_values, cmd_func, all_commands_help=None, **kargs):
        """Create a functional command.

    Args:
      name:        Name of command
      flag_values: FlagValues() instance that needs to be passed as flag_values
                   parameter to any flags registering call.
      cmd_func:    Function to call when command is to be executed.
    """
        Cmd.__init__(self, name, flag_values, **kargs)
        self._all_commands_help = all_commands_help
        self._cmd_func = cmd_func

    def CommandGetHelp(self, unused_argv, cmd_names=None):
        """Get help for command.

    Args:
      unused_argv: Remaining command line flags and arguments after parsing
                   command (that is a copy of sys.argv at the time of the
                   function call with all parsed flags removed); unused in this
                   implementation.
      cmd_names:   By default, if help is being shown for more than one command,
                   and this command defines _all_commands_help, then
                   _all_commands_help will be displayed instead of the class
                   doc. cmd_names is used to determine the number of commands
                   being displayed and if only a single command is display then
                   the class doc is returned.

    Returns:
      __doc__ property for command function or a message stating there is no
      help.
    """
        if type(cmd_names) is list and len(cmd_names) > 1 and (self._all_commands_help is not None):
            return flags.DocToHelp(self._all_commands_help)
        if self._cmd_func.__doc__ is not None:
            return flags.DocToHelp(self._cmd_func.__doc__)
        else:
            return 'No help available'

    def Run(self, argv):
        """Execute the command with given arguments.

    Args:
      argv: Remaining command line flags and arguments after parsing command
            (that is a copy of sys.argv at the time of the function call with
            all parsed flags removed).

    Returns:
      Command return value.
    """
        return self._cmd_func(argv)