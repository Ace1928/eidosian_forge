import os
import re
from .._core import SHELL_NAMES, ShellDetectionFailure
from . import proc, ps
def _get_shell(cmd, *args):
    if cmd.startswith('-'):
        return _get_login_shell(cmd)
    name = os.path.basename(cmd).lower()
    if name == 'rosetta' or QEMU_BIN_REGEX.fullmatch(name):
        cmd = args[0]
        args = args[1:]
        name = os.path.basename(cmd).lower()
    if name in SHELL_NAMES:
        return (name, cmd)
    shell = _get_interpreter_shell(name, args)
    if shell:
        return shell
    return None