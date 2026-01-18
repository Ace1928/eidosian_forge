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
class TestDistutilsSupport(TestCase):

    def setUp(self):
        super().setUp()
        self.skip_if_no_external_compiler()
        unset_macosx_deployment_target()
        self.tmpdir = temp_directory('test_pycc_distutils')
        source_dir = os.path.join(base_path, 'pycc_distutils_usecase')
        self.usecase_dir = os.path.join(self.tmpdir, 'work')
        shutil.copytree(source_dir, self.usecase_dir)

    def check_setup_py(self, setup_py_file):
        import numba
        numba_path = os.path.abspath(os.path.dirname(os.path.dirname(numba.__file__)))
        env = dict(os.environ)
        if env.get('PYTHONPATH', ''):
            env['PYTHONPATH'] = numba_path + os.pathsep + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = numba_path

        def run_python(args):
            p = subprocess.Popen([sys.executable] + args, cwd=self.usecase_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            out, _ = p.communicate()
            rc = p.wait()
            if rc != 0:
                self.fail('python failed with the following output:\n%s' % out.decode('utf-8', 'ignore'))
        run_python([setup_py_file, 'build_ext', '--inplace'])
        code = 'if 1:\n            import pycc_compiled_module as lib\n            assert lib.get_const() == 42\n            res = lib.ones(3)\n            assert list(res) == [1.0, 1.0, 1.0]\n            '
        run_python(['-c', code])

    def check_setup_nested_py(self, setup_py_file):
        import numba
        numba_path = os.path.abspath(os.path.dirname(os.path.dirname(numba.__file__)))
        env = dict(os.environ)
        if env.get('PYTHONPATH', ''):
            env['PYTHONPATH'] = numba_path + os.pathsep + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = numba_path

        def run_python(args):
            p = subprocess.Popen([sys.executable] + args, cwd=self.usecase_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            out, _ = p.communicate()
            rc = p.wait()
            if rc != 0:
                self.fail('python failed with the following output:\n%s' % out.decode('utf-8', 'ignore'))
        run_python([setup_py_file, 'build_ext', '--inplace'])
        code = 'if 1:\n            import nested.pycc_compiled_module as lib\n            assert lib.get_const() == 42\n            res = lib.ones(3)\n            assert list(res) == [1.0, 1.0, 1.0]\n            '
        run_python(['-c', code])

    def test_setup_py_distutils(self):
        self.check_setup_py('setup_distutils.py')

    def test_setup_py_distutils_nested(self):
        self.check_setup_nested_py('setup_distutils_nested.py')

    def test_setup_py_setuptools(self):
        self.check_setup_py('setup_setuptools.py')

    def test_setup_py_setuptools_nested(self):
        self.check_setup_nested_py('setup_setuptools_nested.py')