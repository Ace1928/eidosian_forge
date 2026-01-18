from __future__ import annotations
import importlib.machinery
import importlib.util
import inspect
import marshal
import os
import struct
import sys
from importlib.machinery import ModuleSpec
from types import CodeType, ModuleType
from typing import Any
from coverage import env
from coverage.exceptions import CoverageException, _ExceptionDuringRun, NoCode, NoSource
from coverage.files import canonical_filename, python_reported_file
from coverage.misc import isolate_module
from coverage.python import get_python_source
class PyRunner:
    """Multi-stage execution of Python code.

    This is meant to emulate real Python execution as closely as possible.

    """

    def __init__(self, args: list[str], as_module: bool=False) -> None:
        self.args = args
        self.as_module = as_module
        self.arg0 = args[0]
        self.package: str | None = None
        self.modulename: str | None = None
        self.pathname: str | None = None
        self.loader: DummyLoader | None = None
        self.spec: ModuleSpec | None = None

    def prepare(self) -> None:
        """Set sys.path properly.

        This needs to happen before any importing, and without importing anything.
        """
        path0: str | None
        if self.as_module:
            path0 = os.getcwd()
        elif os.path.isdir(self.arg0):
            path0 = self.arg0
        else:
            path0 = os.path.abspath(os.path.dirname(self.arg0))
        if os.path.isdir(sys.path[0]):
            top_file = inspect.stack()[-1][0].f_code.co_filename
            sys_path_0_abs = os.path.abspath(sys.path[0])
            top_file_dir_abs = os.path.abspath(os.path.dirname(top_file))
            sys_path_0_abs = canonical_filename(sys_path_0_abs)
            top_file_dir_abs = canonical_filename(top_file_dir_abs)
            if sys_path_0_abs != top_file_dir_abs:
                path0 = None
        elif sys.path[1] == os.path.dirname(sys.path[0]):
            del sys.path[1]
        if path0 is not None:
            sys.path[0] = python_reported_file(path0)

    def _prepare2(self) -> None:
        """Do more preparation to run Python code.

        Includes finding the module to run and adjusting sys.argv[0].
        This method is allowed to import code.

        """
        if self.as_module:
            self.modulename = self.arg0
            pathname, self.package, self.spec = find_module(self.modulename)
            if self.spec is not None:
                self.modulename = self.spec.name
            self.loader = DummyLoader(self.modulename)
            assert pathname is not None
            self.pathname = os.path.abspath(pathname)
            self.args[0] = self.arg0 = self.pathname
        elif os.path.isdir(self.arg0):
            for ext in ['.py', '.pyc', '.pyo']:
                try_filename = os.path.join(self.arg0, '__main__' + ext)
                if env.PYVERSION >= (3, 8, 10):
                    try_filename = os.path.abspath(try_filename)
                if os.path.exists(try_filename):
                    self.arg0 = try_filename
                    break
            else:
                raise NoSource(f"Can't find '__main__' module in '{self.arg0}'")
            try_filename = python_reported_file(try_filename)
            self.spec = importlib.machinery.ModuleSpec('__main__', None, origin=try_filename)
            self.spec.has_location = True
            self.package = ''
            self.loader = DummyLoader('__main__')
        else:
            self.loader = DummyLoader('__main__')
        self.arg0 = python_reported_file(self.arg0)

    def run(self) -> None:
        """Run the Python code!"""
        self._prepare2()
        main_mod = ModuleType('__main__')
        from_pyc = self.arg0.endswith(('.pyc', '.pyo'))
        main_mod.__file__ = self.arg0
        if from_pyc:
            main_mod.__file__ = main_mod.__file__[:-1]
        if self.package is not None:
            main_mod.__package__ = self.package
        main_mod.__loader__ = self.loader
        if self.spec is not None:
            main_mod.__spec__ = self.spec
        main_mod.__builtins__ = sys.modules['builtins']
        sys.modules['__main__'] = main_mod
        sys.argv = self.args
        try:
            if from_pyc:
                code = make_code_from_pyc(self.arg0)
            else:
                code = make_code_from_py(self.arg0)
        except CoverageException:
            raise
        except Exception as exc:
            msg = f"Couldn't run '{self.arg0}' as Python code: {exc.__class__.__name__}: {exc}"
            raise CoverageException(msg) from exc
        cwd = os.getcwd()
        try:
            exec(code, main_mod.__dict__)
        except SystemExit:
            raise
        except Exception:
            typ, err, tb = sys.exc_info()
            assert typ is not None
            assert err is not None
            assert tb is not None
            getattr(err, '__context__', None)
            try:
                assert err.__traceback__ is not None
                err.__traceback__ = err.__traceback__.tb_next
                sys.excepthook(typ, err, tb.tb_next)
            except SystemExit:
                raise
            except Exception as exc:
                sys.stderr.write('Error in sys.excepthook:\n')
                typ2, err2, tb2 = sys.exc_info()
                assert typ2 is not None
                assert err2 is not None
                assert tb2 is not None
                err2.__suppress_context__ = True
                assert err2.__traceback__ is not None
                err2.__traceback__ = err2.__traceback__.tb_next
                sys.__excepthook__(typ2, err2, tb2.tb_next)
                sys.stderr.write('\nOriginal exception was:\n')
                raise _ExceptionDuringRun(typ, err, tb.tb_next) from exc
            else:
                sys.exit(1)
        finally:
            os.chdir(cwd)