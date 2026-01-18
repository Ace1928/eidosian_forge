import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
def _drive(self):
    """This function sets up the caputured gdb instance"""
    assert os.path.isfile(self._file_name)
    cmd = [self._gdb_binary, '--interpreter', 'mi']
    if self._init_cmds is not None:
        cmd += list(self._init_cmds)
    cmd += ['--args', self._python, self._file_name]
    self._captured = pexpect.spawn(' '.join(cmd))
    if self._debug:
        self._captured.logfile = sys.stdout.buffer