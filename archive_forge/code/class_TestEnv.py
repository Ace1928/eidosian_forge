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
class TestEnv(TestCase):

    def test_env(self):
        env = _ip.run_line_magic('env', '')
        self.assertTrue(isinstance(env, dict))

    def test_env_secret(self):
        env = _ip.run_line_magic('env', '')
        hidden = '<hidden>'
        with mock.patch.dict(os.environ, {'API_KEY': 'abc123', 'SECRET_THING': 'ssshhh', 'JUPYTER_TOKEN': '', 'VAR': 'abc'}):
            env = _ip.run_line_magic('env', '')
        assert env['API_KEY'] == hidden
        assert env['SECRET_THING'] == hidden
        assert env['JUPYTER_TOKEN'] == hidden
        assert env['VAR'] == 'abc'

    def test_env_get_set_simple(self):
        env = _ip.run_line_magic('env', 'var val1')
        self.assertEqual(env, None)
        self.assertEqual(os.environ['var'], 'val1')
        self.assertEqual(_ip.run_line_magic('env', 'var'), 'val1')
        env = _ip.run_line_magic('env', 'var=val2')
        self.assertEqual(env, None)
        self.assertEqual(os.environ['var'], 'val2')

    def test_env_get_set_complex(self):
        env = _ip.run_line_magic('env', "var 'val1 '' 'val2")
        self.assertEqual(env, None)
        self.assertEqual(os.environ['var'], "'val1 '' 'val2")
        self.assertEqual(_ip.run_line_magic('env', 'var'), "'val1 '' 'val2")
        env = _ip.run_line_magic('env', 'var=val2 val3="val4')
        self.assertEqual(env, None)
        self.assertEqual(os.environ['var'], 'val2 val3="val4')

    def test_env_set_bad_input(self):
        self.assertRaises(UsageError, lambda: _ip.run_line_magic('set_env', 'var'))

    def test_env_set_whitespace(self):
        self.assertRaises(UsageError, lambda: _ip.run_line_magic('env', 'var A=B'))