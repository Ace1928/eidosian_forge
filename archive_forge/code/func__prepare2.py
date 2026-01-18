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