import os
import sys
import platform
import inspect
import traceback
import pdb
import re
import linecache
import time
from fnmatch import fnmatch
from timeit import default_timer as clock
import doctest as pdoctest  # avoid clashing with our doctest() function
from doctest import DocTestFinder, DocTestRunner
import random
import subprocess
import shutil
import signal
import stat
import tempfile
import warnings
from contextlib import contextmanager
from inspect import unwrap
from sympy.core.cache import clear_cache
from sympy.external import import_module
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from collections import namedtuple
class SymPyDocTests:

    def __init__(self, reporter, normal):
        self._count = 0
        self._root_dir = get_sympy_dir()
        self._reporter = reporter
        self._reporter.root_dir(self._root_dir)
        self._normal = normal
        self._testfiles = []

    def test(self):
        """
        Runs the tests and returns True if all tests pass, otherwise False.
        """
        self._reporter.start()
        for f in self._testfiles:
            try:
                self.test_file(f)
            except KeyboardInterrupt:
                print(' interrupted by user')
                self._reporter.finish()
                raise
        return self._reporter.finish()

    def test_file(self, filename):
        clear_cache()
        from io import StringIO
        import sympy.interactive.printing as interactive_printing
        from sympy.printing.pretty.pretty import pprint_use_unicode
        rel_name = filename[len(self._root_dir) + 1:]
        dirname, file = os.path.split(filename)
        module = rel_name.replace(os.sep, '.')[:-3]
        if rel_name.startswith('examples'):
            sys.path.insert(0, dirname)
            module = file[:-3]
        try:
            module = pdoctest._normalize_module(module)
            tests = SymPyDocTestFinder().find(module)
        except (SystemExit, KeyboardInterrupt):
            raise
        except ImportError:
            self._reporter.import_error(filename, sys.exc_info())
            return
        finally:
            if rel_name.startswith('examples'):
                del sys.path[0]
        tests = [test for test in tests if len(test.examples) > 0]
        tests.sort(key=lambda x: -x.lineno)
        if not tests:
            return
        self._reporter.entering_filename(filename, len(tests))
        for test in tests:
            assert len(test.examples) != 0
            if self._reporter._verbose:
                self._reporter.write('\n{} '.format(test.name))
            if '_doctest_depends_on' in test.globs:
                try:
                    self._check_dependencies(**test.globs['_doctest_depends_on'])
                except DependencyError as e:
                    self._reporter.test_skip(v=str(e))
                    continue
            runner = SymPyDocTestRunner(verbose=self._reporter._verbose == 2, optionflags=pdoctest.ELLIPSIS | pdoctest.NORMALIZE_WHITESPACE | pdoctest.IGNORE_EXCEPTION_DETAIL)
            runner._checker = SymPyOutputChecker()
            old = sys.stdout
            new = old if self._reporter._verbose == 2 else StringIO()
            sys.stdout = new
            if not self._normal:
                test.globs = {}
            old_displayhook = sys.displayhook
            use_unicode_prev = setup_pprint()
            try:
                f, t = runner.run(test, out=new.write, clear_globs=False)
            except KeyboardInterrupt:
                raise
            finally:
                sys.stdout = old
            if f > 0:
                self._reporter.doctest_fail(test.name, new.getvalue())
            else:
                self._reporter.test_pass()
                sys.displayhook = old_displayhook
                interactive_printing.NO_GLOBAL = False
                pprint_use_unicode(use_unicode_prev)
        self._reporter.leaving_filename()

    def get_test_files(self, dir, pat='*.py', init_only=True):
        """
        Returns the list of \\*.py files (default) from which docstrings
        will be tested which are at or below directory ``dir``. By default,
        only those that have an __init__.py in their parent directory
        and do not start with ``test_`` will be included.
        """

        def importable(x):
            """
            Checks if given pathname x is an importable module by checking for
            __init__.py file.

            Returns True/False.

            Currently we only test if the __init__.py file exists in the
            directory with the file "x" (in theory we should also test all the
            parent dirs).
            """
            init_py = os.path.join(os.path.dirname(x), '__init__.py')
            return os.path.exists(init_py)
        dir = os.path.join(self._root_dir, convert_to_native_paths([dir])[0])
        g = []
        for path, folders, files in os.walk(dir):
            g.extend([os.path.join(path, f) for f in files if not f.startswith('test_') and fnmatch(f, pat)])
        if init_only:
            g = [x for x in g if importable(x)]
        return [os.path.normcase(gi) for gi in g]

    def _check_dependencies(self, executables=(), modules=(), disable_viewers=(), python_version=(3, 5)):
        """
        Checks if the dependencies for the test are installed.

        Raises ``DependencyError`` it at least one dependency is not installed.
        """
        for executable in executables:
            if not shutil.which(executable):
                raise DependencyError('Could not find %s' % executable)
        for module in modules:
            if module == 'matplotlib':
                matplotlib = import_module('matplotlib', import_kwargs={'fromlist': ['pyplot', 'cm', 'collections']}, min_module_version='1.0.0', catch=(RuntimeError,))
                if matplotlib is None:
                    raise DependencyError('Could not import matplotlib')
            elif not import_module(module):
                raise DependencyError('Could not import %s' % module)
        if disable_viewers:
            tempdir = tempfile.mkdtemp()
            os.environ['PATH'] = '%s:%s' % (tempdir, os.environ['PATH'])
            vw = '#!/usr/bin/env python3\nimport sys\nif len(sys.argv) <= 1:\n    exit("wrong number of args")\n'
            for viewer in disable_viewers:
                with open(os.path.join(tempdir, viewer), 'w') as fh:
                    fh.write(vw)
                os.chmod(os.path.join(tempdir, viewer), stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)
        if python_version:
            if sys.version_info < python_version:
                raise DependencyError('Requires Python >= ' + '.'.join(map(str, python_version)))
        if 'pyglet' in modules:
            import pyglet

            class DummyWindow:

                def __init__(self, *args, **kwargs):
                    self.has_exit = True
                    self.width = 600
                    self.height = 400

                def set_vsync(self, x):
                    pass

                def switch_to(self):
                    pass

                def push_handlers(self, x):
                    pass

                def close(self):
                    pass
            pyglet.window.Window = DummyWindow