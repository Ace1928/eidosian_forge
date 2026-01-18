from __future__ import print_function, absolute_import
import os
import tempfile
import unittest
import sys
import re
import warnings
import io
from textwrap import dedent
from future.utils import bind_method, PY26, PY3, PY2, PY27
from future.moves.subprocess import check_output, STDOUT, CalledProcessError
def _read_test_script(self, filename='mytestscript.py'):
    with io.open(self.tempdir + filename, 'rt', encoding='utf-8') as f:
        newsource = f.read()
    return newsource