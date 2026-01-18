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
@staticmethod
def _run_with_timing(run, nruns):
    """
        Run function `run` and print timing information.

        Parameters
        ----------
        run : callable
            Any callable object which takes no argument.
        nruns : int
            Number of times to execute `run`.

        """
    twall0 = time.perf_counter()
    if nruns == 1:
        t0 = clock2()
        run()
        t1 = clock2()
        t_usr = t1[0] - t0[0]
        t_sys = t1[1] - t0[1]
        print('\nIPython CPU timings (estimated):')
        print('  User   : %10.2f s.' % t_usr)
        print('  System : %10.2f s.' % t_sys)
    else:
        runs = range(nruns)
        t0 = clock2()
        for nr in runs:
            run()
        t1 = clock2()
        t_usr = t1[0] - t0[0]
        t_sys = t1[1] - t0[1]
        print('\nIPython CPU timings (estimated):')
        print('Total runs performed:', nruns)
        print('  Times  : %10s   %10s' % ('Total', 'Per run'))
        print('  User   : %10.2f s, %10.2f s.' % (t_usr, t_usr / nruns))
        print('  System : %10.2f s, %10.2f s.' % (t_sys, t_sys / nruns))
    twall1 = time.perf_counter()
    print('Wall time: %10.2f s.' % (twall1 - twall0))