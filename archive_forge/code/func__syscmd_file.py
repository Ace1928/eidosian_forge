import collections
import os
import re
import sys
import functools
import itertools
def _syscmd_file(target, default=''):
    """ Interface to the system's file command.

        The function uses the -b option of the file command to have it
        omit the filename in its output. Follow the symlinks. It returns
        default in case the command should fail.

    """
    if sys.platform in ('dos', 'win32', 'win16'):
        return default
    try:
        import subprocess
    except ImportError:
        return default
    target = _follow_symlinks(target)
    env = dict(os.environ, LC_ALL='C')
    try:
        output = subprocess.check_output(['file', '-b', target], stderr=subprocess.DEVNULL, env=env)
    except (OSError, subprocess.CalledProcessError):
        return default
    if not output:
        return default
    return output.decode('latin-1')