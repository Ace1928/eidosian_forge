import gc
import io
import os
import re
import shlex
import sys
import warnings
from importlib import invalidate_caches
from io import StringIO
from pathlib import Path
from textwrap import dedent
from unittest import TestCase, mock
import pytest
from IPython import get_ipython
from IPython.core import magic
from IPython.core.error import UsageError
from IPython.core.magic import (
from IPython.core.magics import code, execution, logging, osm, script
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
from IPython.utils.process import find_cmd
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.syspathcontext import prepended_to_syspath
from .test_debugger import PdbTestInput
from tempfile import NamedTemporaryFile
from IPython.core.magic import (
def _run_edit_test(arg_s, exp_filename=None, exp_lineno=-1, exp_contents=None, exp_is_temp=None):
    ip = get_ipython()
    M = code.CodeMagics(ip)
    last_call = ['', '']
    opts, args = M.parse_options(arg_s, 'prxn:')
    filename, lineno, is_temp = M._find_edit_target(ip, args, opts, last_call)
    if exp_filename is not None:
        assert exp_filename == filename
    if exp_contents is not None:
        with io.open(filename, 'r', encoding='utf-8') as f:
            contents = f.read()
        assert exp_contents == contents
    if exp_lineno != -1:
        assert exp_lineno == lineno
    if exp_is_temp is not None:
        assert exp_is_temp == is_temp