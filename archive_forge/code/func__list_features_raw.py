import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
def _list_features_raw(self):
    cmd = '-list-features'
    self._run_command(cmd, expect='\\^done,.*\\r\\n')