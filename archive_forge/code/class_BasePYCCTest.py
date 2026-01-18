import contextlib
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import skip
from ctypes import *
import numpy as np
import llvmlite.binding as ll
from numba.core import utils
from numba.tests.support import (TestCase, tag, import_dynamic, temp_directory,
import unittest
class BasePYCCTest(TestCase):

    def setUp(self):
        unset_macosx_deployment_target()
        self.tmpdir = temp_directory('test_pycc')
        tempfile.tempdir = self.tmpdir

    def tearDown(self):
        tempfile.tempdir = None
        from numba.pycc.decorators import clear_export_registry
        clear_export_registry()

    @contextlib.contextmanager
    def check_c_ext(self, extdir, name):
        sys.path.append(extdir)
        try:
            lib = import_dynamic(name)
            yield lib
        finally:
            sys.path.remove(extdir)
            sys.modules.pop(name, None)