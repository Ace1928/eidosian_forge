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

        Simple magic to quickly define a code transformer for all IPython's future imput.

        ``__code__`` and ``__ret__`` are special variable that represent the code to run
        and the value of the last expression of ``__code__`` respectively.

        Examples
        --------

        .. ipython::

            In [1]: %%code_wrap before_after
               ...: print('before')
               ...: __code__
               ...: print('after')
               ...: __ret__


            In [2]: 1
            before
            after
            Out[2]: 1

            In [3]: %code_wrap --list
            before_after

            In [4]: %code_wrap --list-all
            before_after :
                print('before')
                __code__
                print('after')
                __ret__

            In [5]: %code_wrap --remove before_after

        