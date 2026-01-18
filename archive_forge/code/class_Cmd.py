from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
class Cmd(object):
    """Abstract class describing and implementing a command.

  When creating code for a command, at least you have to derive this class
  and override method Run(). The other methods of this class might be
  overridden as well. Check their documentation for details. If the command
  needs any specific flags, use __init__ for registration.
  """

    def __init__(self, name, flag_values, command_aliases=None):
        """Initialize and check whether self is actually a Cmd instance.

    This can be used to register command specific flags. If you do so
    remember that you have to provide the 'flag_values=flag_values'
    parameter to any flags.DEFINE_*() call.

    Args:
      name:            Name of the command
      flag_values:     FlagValues() instance that needs to be passed as
                       flag_values parameter to any flags registering call.
      command_aliases: A list of command aliases that the command can be run as.
    Raises:
      AppCommandsError: if self is Cmd (Cmd is abstract)
    """
        self._command_name = name
        self._command_aliases = command_aliases
        self._command_flags = flag_values
        self._all_commands_help = None
        if type(self) is Cmd:
            raise AppCommandsError('Cmd is abstract and cannot be instantiated')

    def Run(self, argv):
        """Execute the command. Must be provided by the implementing class.

    Args:
      argv: Remaining command line arguments after parsing flags and command
            (that is a copy of sys.argv at the time of the function call with
            all parsed flags removed).

    Returns:
      0 for success, anything else for failure (must return with integer).
      Alternatively you may return None (or not use a return statement at all).

    Raises:
      AppCommandsError: Always as in must be overwritten
    """
        raise AppCommandsError('%s.%s.Run() is not implemented' % (type(self).__module__, type(self).__name__))

    def CommandRun(self, argv):
        """Execute the command with given arguments.

    First register and parse additional flags. Then run the command.

    Returns:
      Command return value.

    Args:
      argv: Remaining command line arguments after parsing command and flags
            (that is a copy of sys.argv at the time of the function call with
            all parsed flags removed).
    """
        FLAGS.AppendFlagValues(self._command_flags)
        orig_app_usage = app.usage

        def ReplacementAppUsage(shorthelp=0, writeto_stdout=1, detailed_error=None, exitcode=None):
            AppcommandsUsage(shorthelp, writeto_stdout, detailed_error, exitcode=1, show_cmd=self._command_name, show_global_flags=True)
        app.usage = ReplacementAppUsage
        try:
            try:
                argv = ParseFlagsWithUsage(argv)
                if FLAGS.run_with_pdb:
                    ret = pdb.runcall(self.Run, argv)
                else:
                    ret = self.Run(argv)
                if ret is None:
                    ret = 0
                else:
                    assert isinstance(ret, int)
                return ret
            except app.UsageError as error:
                app.usage(shorthelp=1, detailed_error=error, exitcode=error.exitcode)
        finally:
            app.usage = orig_app_usage
            for flag_name in self._command_flags.FlagDict():
                delattr(FLAGS, flag_name)

    def CommandGetHelp(self, unused_argv, cmd_names=None):
        """Get help string for command.

    Args:
      unused_argv: Remaining command line flags and arguments after parsing
                   command (that is a copy of sys.argv at the time of the
                   function call with all parsed flags removed); unused in this
                   default implementation, but may be used in subclasses.
      cmd_names:   Complete list of commands for which help is being shown at
                   the same time. This is used to determine whether to return
                   _all_commands_help, or the command's docstring.
                   (_all_commands_help is used, if not None, when help is being
                   shown for more than one command, otherwise the command's
                   docstring is used.)

    Returns:
      Help string, one of the following (by order):
        - Result of the registered 'help' function (if any)
        - Doc string of the Cmd class (if any)
        - Default fallback string
    """
        if type(cmd_names) is list and len(cmd_names) > 1 and (self._all_commands_help is not None):
            return flags.DocToHelp(self._all_commands_help)
        elif self.__doc__:
            return flags.DocToHelp(self.__doc__)
        else:
            return 'No help available'

    def CommandGetAliases(self):
        """Get aliases for command.

    Returns:
      aliases: list of aliases for the command.
    """
        return self._command_aliases