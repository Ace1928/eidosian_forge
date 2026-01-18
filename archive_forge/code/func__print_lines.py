import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def _print_lines(self, lines, start, breaks=(), frame=None):
    """Print a range of lines."""
    if frame:
        current_lineno = frame.f_lineno
        exc_lineno = self.tb_lineno.get(frame, -1)
    else:
        current_lineno = exc_lineno = -1
    for lineno, line in enumerate(lines, start):
        s = str(lineno).rjust(3)
        if len(s) < 4:
            s += ' '
        if lineno in breaks:
            s += 'B'
        else:
            s += ' '
        if lineno == current_lineno:
            s += '->'
        elif lineno == exc_lineno:
            s += '>>'
        self.message(s + '\t' + line.rstrip())