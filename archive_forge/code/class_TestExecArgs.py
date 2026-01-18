import errno
import os
import pty
import re
import select
import subprocess
import sys
import tempfile
import unittest
from textwrap import dedent
from bpython import args
from bpython.config import getpreferredencoding
from bpython.test import FixLanguageTestCase as TestCase
class TestExecArgs(unittest.TestCase):

    def test_exec_dunder_file(self):
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(dedent('                import sys\n                sys.stderr.write(__file__)\n                sys.stderr.flush()'))
            f.flush()
            _, stderr = run_with_tty([sys.executable] + ['-m', 'bpython.curtsies', f.name])
            self.assertEqual(stderr.strip(), f.name)

    def test_exec_nonascii_file(self):
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(dedent('                # coding: utf-8\n                "你好 # nonascii"\n                '))
            f.flush()
            _, stderr = run_with_tty([sys.executable, '-m', 'bpython.curtsies', f.name])
            self.assertEqual(len(stderr), 0)

    def test_exec_nonascii_file_linenums(self):
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(dedent('                1/0\n                '))
            f.flush()
            _, stderr = run_with_tty([sys.executable, '-m', 'bpython.curtsies', f.name])
            self.assertIn('line 1', clean_colors(stderr))