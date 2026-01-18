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
@needs_setuptools
class TestCompilerChecks(TestCase):

    @_windows_only
    def test_windows_compiler_validity(self):
        from numba.pycc.platform import external_compiler_works
        is_running_conda_build = os.environ.get('CONDA_BUILD', None) is not None
        if is_running_conda_build:
            if os.environ.get('VSINSTALLDIR', None) is not None:
                self.assertTrue(external_compiler_works())