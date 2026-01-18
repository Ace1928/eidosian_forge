import inspect
import locale
import logging
import logging.handlers
import os
import sys
from cliff import _argparse
from . import complete
from . import help
from . import utils
def _set_streams(self, stdin, stdout, stderr):
    try:
        locale.setlocale(locale.LC_ALL, '')
    except locale.Error:
        pass
    self.stdin = stdin or sys.stdin
    self.stdout = stdout or sys.stdout
    self.stderr = stderr or sys.stderr