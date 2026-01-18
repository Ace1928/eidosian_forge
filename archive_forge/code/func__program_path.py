import os
import re
import shutil
import sys
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