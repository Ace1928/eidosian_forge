import os
import re
import shutil
import sys
class KillFilter(CommandFilter):
    """Specific filter for the kill calls.

       1st argument is the user to run /bin/kill under
       2nd argument is the location of the affected executable
           if the argument is not absolute, it is checked against $PATH
       Subsequent arguments list the accepted signals (if any)

       This filter relies on /proc to accurately determine affected
       executable, so it will only work on procfs-capable systems (not OSX).
    """

    def __init__(self, *args):
        super(KillFilter, self).__init__('/bin/kill', *args)

    @staticmethod
    def _program_path(command):
        """Try to determine the full path for command.

        Return command if the full path cannot be found.
        """
        if hasattr(shutil, 'which'):
            return shutil.which(command)
        if os.path.isabs(command):
            return command
        path = os.environ.get('PATH', os.defpath).split(os.pathsep)
        for dir in path:
            program = os.path.join(dir, command)
            if os.path.isfile(program):
                return program
        return command

    def _program(self, pid):
        """Determine the program associated with pid"""
        try:
            command = os.readlink('/proc/%d/exe' % int(pid))
        except (ValueError, EnvironmentError):
            return None
        command = command.partition('\x00')[0]
        if command.endswith(' (deleted)'):
            command = command[:-len(' (deleted)')]
        if os.path.isfile(command):
            return command
        try:
            with open('/proc/%d/cmdline' % int(pid)) as pfile:
                cmdline = pfile.read().partition('\x00')[0]
            cmdline = self._program_path(cmdline)
            if os.path.isfile(cmdline):
                command = cmdline
            return command
        except EnvironmentError:
            return None

    def match(self, userargs):
        if not userargs or userargs[0] != 'kill':
            return False
        args = list(userargs)
        if len(args) == 3:
            signal = args.pop(1)
            if signal not in self.args[1:]:
                return False
        else:
            if len(args) != 2:
                return False
            if len(self.args) > 1:
                return False
        command = self._program(args[1])
        if not command:
            return False
        kill_command = self.args[0]
        if os.path.isabs(kill_command):
            return kill_command == command
        return os.path.isabs(command) and kill_command == os.path.basename(command) and (os.path.dirname(command) in os.environ.get('PATH', '').split(':'))