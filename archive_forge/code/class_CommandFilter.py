import os
import re
import shutil
import sys
class CommandFilter(object):
    """Command filter only checking that the 1st argument matches exec_path."""

    def __init__(self, exec_path, run_as, *args):
        self.name = ''
        self.exec_path = exec_path
        self.run_as = run_as
        self.args = args
        self.real_exec = None

    def get_exec(self, exec_dirs=None):
        """Returns existing executable, or empty string if none found."""
        exec_dirs = exec_dirs or []
        if self.real_exec is not None:
            return self.real_exec
        if os.path.isabs(self.exec_path):
            if os.access(self.exec_path, os.X_OK):
                self.real_exec = self.exec_path
        else:
            for binary_path in exec_dirs:
                expanded_path = os.path.join(binary_path, self.exec_path)
                if os.access(expanded_path, os.X_OK):
                    self.real_exec = expanded_path
                    break
        return self.real_exec

    def match(self, userargs):
        """Only check that the first argument (command) matches exec_path."""
        if userargs:
            base_path_matches = os.path.basename(self.exec_path) == userargs[0]
            exact_path_matches = self.exec_path == userargs[0]
            return exact_path_matches or base_path_matches
        return False

    def preexec(self):
        """Setuid in subprocess right before command is invoked."""
        if self.run_as != 'root':
            os.setuid(_getuid(self.run_as))

    def get_command(self, userargs, exec_dirs=None):
        """Returns command to execute."""
        exec_dirs = exec_dirs or []
        to_exec = self.get_exec(exec_dirs=exec_dirs) or self.exec_path
        return [to_exec] + userargs[1:]

    def get_environment(self, userargs):
        """Returns specific environment to set, None if none."""
        return None