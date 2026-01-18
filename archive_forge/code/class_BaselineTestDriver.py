import enum
import glob
import logging
import math
import os
import operator
import re
import subprocess
import sys
from io import StringIO
from unittest import *
import unittest as _unittest
import pytest as pytest
from pyomo.common.collections import Mapping, Sequence
from pyomo.common.dependencies import attempt_import, check_min_version
from pyomo.common.errors import InvalidValueError
from pyomo.common.fileutils import import_file
from pyomo.common.log import LoggingIntercept, pyomo_formatter
from pyomo.common.tee import capture_output
from unittest import mock
class BaselineTestDriver(object):
    """Generic driver for performing baseline tests in bulk

    This test driver was originally crafted for testing the examples in
    the Pyomo Book, and has since been generalized to reuse in testing
    ".. literalinclude:" examples from the Online Docs.

    We expect that consumers of this class will derive from both this
    class and `pyomo.common.unittest.TestCase`, and then use
    `parameterized` to declare tests that call either the
    :py:meth:`python_test_driver` or :py:meth:`shell_test_driver`
    methods.

    Note that derived classes must declare two class attributes:

    Class Attributes
    ----------------
    solver_dependencies: Dict[str, List[str]]

        maps the test name to a list of required solvers.  If any solver
        is not available, then the test will be skipped.

    package_dependencies: Dict[str, List[str]]

        maps the test name to a list of required modules.  If any module
        is not available, then the test will be skipped.

    """

    @staticmethod
    def custom_name_func(test_func, test_num, test_params):
        func_name = test_func.__name__
        return 'test_%s_%s' % (test_params.args[0], func_name[-2:])

    def __init__(self, test):
        if getattr(self.__class__, 'solver_available', None) is None:
            self.initialize_dependencies()
        super().__init__(test)

    def initialize_dependencies(self):
        from pyomo.opt import check_available_solvers
        cls = self.__class__
        solvers_used = set(sum(list(cls.solver_dependencies.values()), []))
        available_solvers = check_available_solvers(*solvers_used)
        cls.solver_available = {solver_: solver_ in available_solvers for solver_ in solvers_used}
        cls.package_available = {}
        cls.package_modules = {}
        packages_used = set(sum(list(cls.package_dependencies.values()), []))
        for package_ in packages_used:
            pack, pack_avail = attempt_import(package_, defer_check=False)
            cls.package_available[package_] = pack_avail
            cls.package_modules[package_] = pack

    @classmethod
    def _find_tests(cls, test_dirs, pattern):
        test_tuples = []
        for testdir in test_dirs:
            for fname in list(glob.glob(os.path.join(testdir, pattern))) + list(glob.glob(os.path.join(testdir, '*', pattern))):
                test_file = os.path.abspath(fname)
                bname = os.path.basename(test_file)
                dir_ = os.path.dirname(test_file)
                name = os.path.splitext(bname)[0]
                tname = os.path.basename(dir_) + '_' + name
                suffix = None
                for suffix_ in ['.txt', '.yml']:
                    if os.path.exists(os.path.join(dir_, name + suffix_)):
                        suffix = suffix_
                        break
                if suffix is not None:
                    tname = tname.replace('-', '_')
                    tname = tname.replace('.', '_')
                    test_tuples.append((tname, test_file, os.path.join(dir_, name + suffix)))
        test_tuples.sort()
        return test_tuples

    @classmethod
    def gather_tests(cls, test_dirs):
        sh_test_tuples = cls._find_tests(test_dirs, '*.sh')
        py_test_tuples = cls._find_tests(test_dirs, '*.py')
        sh_files = set(map(operator.itemgetter(1), sh_test_tuples))
        py_test_tuples = list(filter(lambda t: t[1][:-3] + '.sh' not in sh_files, py_test_tuples))
        return (py_test_tuples, sh_test_tuples)

    def check_skip(self, name):
        """
        Return a boolean if the test should be skipped
        """
        if name in self.solver_dependencies:
            solvers_ = self.solver_dependencies[name]
            if not all([self.solver_available[i] for i in solvers_]):
                _missing = []
                for i in solvers_:
                    if not self.solver_available[i]:
                        _missing.append(i)
                return 'Solver%s %s %s not available' % ('s' if len(_missing) > 1 else '', ', '.join(_missing), 'are' if len(_missing) > 1 else 'is')
        if name in self.package_dependencies:
            packages_ = self.package_dependencies[name]
            if not all([self.package_available[i] for i in packages_]):
                _missing = []
                for i in packages_:
                    if not self.package_available[i]:
                        _missing.append(i)
                return 'Package%s %s %s not available' % ('s' if len(_missing) > 1 else '', ', '.join(_missing), 'are' if len(_missing) > 1 else 'is')
            if 'pandas' in self.package_dependencies[name] and 'xlrd' in self.package_dependencies[name]:
                if check_min_version(self.package_modules['xlrd'], '2.0.1') and (not check_min_version(self.package_modules['pandas'], '1.1.6')):
                    return 'Incompatible versions of xlrd and pandas'
        return False

    def filter_fcn(self, line):
        """
        Ignore certain text when comparing output with baseline
        """
        for field in ('[', 'password:', 'http:', 'Job ', 'Importing module', 'Function', 'File', 'Matplotlib', 'Memory:', '-------', '=======', '    ^'):
            if line.startswith(field):
                return True
        for field in ('Total CPU', 'Ipopt', 'license', 'time:', 'Time:', 'with format cpxlp', 'usermodel = <module', 'execution time=', 'Solver results file:', 'TokenServer', 'function calls', 'List reduced', '.py:', ' {built-in method', ' {method', ' {pyomo.core.expr.numvalue.as_numeric}'):
            if field in line:
                return True
        return False

    def filter_file_contents(self, lines, abstol=None):
        filtered = []
        deprecated = None
        for line in lines:
            if line.startswith('WARNING: DEPRECATED:'):
                deprecated = ''
            if deprecated is not None:
                deprecated += line
                if re.search('\\(called\\s+from[^)]+\\)', deprecated):
                    deprecated = None
                continue
            if not line or self.filter_fcn(line):
                continue
            if 'seconds' in line:
                s = line.find('seconds') + 7
                line = line[s:]
            item_list = []
            items = line.strip().split()
            for i in items:
                if '.inf' in i:
                    i = i.replace('.inf', 'inf')
                if 'null' in i:
                    i = i.replace('null', 'None')
                try:
                    item_list.append(float(i))
                except:
                    item_list.append(i)
            if len(item_list) == 2 and item_list[0] == 'Value:' and (type(item_list[1]) is float) and (abs(item_list[1]) < (abstol or 0)) and (len(filtered[-1]) == 1) and (filtered[-1][0][-1] == ':'):
                filtered.pop()
            else:
                filtered.append(item_list)
        return filtered

    def compare_baseline(self, test_output, baseline, abstol=1e-06, reltol=None):
        out_filtered = self.filter_file_contents(test_output.strip().split('\n'), abstol)
        base_filtered = self.filter_file_contents(baseline.strip().split('\n'), abstol)
        try:
            self.assertStructuredAlmostEqual(out_filtered, base_filtered, abstol=abstol, reltol=reltol, allow_second_superset=False)
            return True
        except self.failureException:
            print('---------------------------------')
            print('BASELINE FILE')
            print('---------------------------------')
            print(baseline)
            print('=================================')
            print('---------------------------------')
            print('TEST OUTPUT FILE')
            print('---------------------------------')
            print(test_output)
            raise

    def python_test_driver(self, tname, test_file, base_file):
        bname = os.path.basename(test_file)
        _dir = os.path.dirname(test_file)
        skip_msg = self.check_skip('test_' + tname)
        if skip_msg:
            raise _unittest.SkipTest(skip_msg)
        with open(base_file, 'r') as FILE:
            baseline = FILE.read()
        cwd = os.getcwd()
        try:
            os.chdir(_dir)
            with capture_output(None, True) as OUT:
                with LoggingIntercept(sys.stdout, formatter=pyomo_formatter):
                    import_file(bname, infer_package=False, module_name='__main__')
        finally:
            os.chdir(cwd)
        try:
            self.compare_baseline(OUT.getvalue(), baseline)
        except:
            if os.environ.get('PYOMO_TEST_UPDATE_BASELINES', None):
                with open(base_file, 'w') as FILE:
                    FILE.write(OUT.getvalue())
            raise

    def shell_test_driver(self, tname, test_file, base_file):
        bname = os.path.basename(test_file)
        _dir = os.path.dirname(test_file)
        skip_msg = self.check_skip('test_' + tname)
        if skip_msg:
            raise _unittest.SkipTest(skip_msg)
        if os.name == 'nt':
            raise _unittest.SkipTest('Shell tests are not runnable on Windows')
        with open(base_file, 'r') as FILE:
            baseline = FILE.read()
        cwd = os.getcwd()
        try:
            os.chdir(_dir)
            _env = os.environ.copy()
            _env['PATH'] = os.pathsep.join([os.path.dirname(sys.executable), _env['PATH']])
            rc = subprocess.run(['bash', bname], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=_dir, env=_env)
        finally:
            os.chdir(cwd)
        try:
            self.compare_baseline(rc.stdout.decode(), baseline)
        except:
            if os.environ.get('PYOMO_TEST_UPDATE_BASELINES', None):
                with open(base_file, 'w') as FILE:
                    FILE.write(rc.stdout.decode())
            raise