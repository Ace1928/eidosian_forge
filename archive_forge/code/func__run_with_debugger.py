import ast
import bdb
import builtins as builtin_mod
import copy
import cProfile as profile
import gc
import itertools
import math
import os
import pstats
import re
import shlex
import sys
import time
import timeit
from typing import Dict, Any
from ast import (
from io import StringIO
from logging import error
from pathlib import Path
from pdb import Restart
from textwrap import dedent, indent
from warnings import warn
from IPython.core import magic_arguments, oinspect, page
from IPython.core.displayhook import DisplayHook
from IPython.core.error import UsageError
from IPython.core.macro import Macro
from IPython.core.magic import (
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.capture import capture_output
from IPython.utils.contexts import preserve_keys
from IPython.utils.ipstruct import Struct
from IPython.utils.module_paths import find_mod
from IPython.utils.path import get_py_filename, shellglob
from IPython.utils.timing import clock, clock2
from IPython.core.magics.ast_mod import ReplaceCodeTransformer
def _run_with_debugger(self, code, code_ns, filename=None, bp_line=None, bp_file=None, local_ns=None):
    """
        Run `code` in debugger with a break point.

        Parameters
        ----------
        code : str
            Code to execute.
        code_ns : dict
            A namespace in which `code` is executed.
        filename : str
            `code` is ran as if it is in `filename`.
        bp_line : int, optional
            Line number of the break point.
        bp_file : str, optional
            Path to the file in which break point is specified.
            `filename` is used if not given.
        local_ns : dict, optional
            A local namespace in which `code` is executed.

        Raises
        ------
        UsageError
            If the break point given by `bp_line` is not valid.

        """
    deb = self.shell.InteractiveTB.pdb
    if not deb:
        self.shell.InteractiveTB.pdb = self.shell.InteractiveTB.debugger_cls()
        deb = self.shell.InteractiveTB.pdb
    if hasattr(deb, 'curframe'):
        del deb.curframe
    bdb.Breakpoint.next = 1
    bdb.Breakpoint.bplist = {}
    bdb.Breakpoint.bpbynumber = [None]
    deb.clear_all_breaks()
    if bp_line is not None:
        maxtries = 10
        bp_file = bp_file or filename
        checkline = deb.checkline(bp_file, bp_line)
        if not checkline:
            for bp in range(bp_line + 1, bp_line + maxtries + 1):
                if deb.checkline(bp_file, bp):
                    break
            else:
                msg = '\nI failed to find a valid line to set a breakpoint\nafter trying up to line: %s.\nPlease set a valid breakpoint manually with the -b option.' % bp
                raise UsageError(msg)
        deb.do_break('%s:%s' % (bp_file, bp_line))
    if filename:
        deb._wait_for_mainpyfile = True
        deb.mainpyfile = deb.canonic(filename)
    print("NOTE: Enter 'c' at the %s prompt to continue execution." % deb.prompt)
    try:
        if filename:
            deb._exec_filename = filename
        while True:
            try:
                trace = sys.gettrace()
                deb.run(code, code_ns, local_ns)
            except Restart:
                print('Restarting')
                if filename:
                    deb._wait_for_mainpyfile = True
                    deb.mainpyfile = deb.canonic(filename)
                continue
            else:
                break
            finally:
                sys.settrace(trace)
    except:
        etype, value, tb = sys.exc_info()
        self.shell.InteractiveTB(etype, value, tb, tb_offset=3)