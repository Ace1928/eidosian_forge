from test import support
from test.support import import_helper
import_helper.import_module('_multiprocessing')
import importlib
import importlib.machinery
import unittest
import sys
import os
import os.path
import py_compile
from test.support import os_helper
from test.support.script_helper import (
import multiprocess as multiprocessing
import_helper.import_module('multiprocess.synchronize')
import sys
import time
from multiprocess import Pool, set_start_method
import sys
import time
from multiprocess import Pool, set_start_method
import sys, os.path, runpy
def _make_test_script(script_dir, script_basename, source=test_source, omit_suffix=False):
    to_return = make_script(script_dir, script_basename, source, omit_suffix)
    if script_basename == 'check_sibling':
        make_script(script_dir, 'sibling', '')
    importlib.invalidate_caches()
    return to_return