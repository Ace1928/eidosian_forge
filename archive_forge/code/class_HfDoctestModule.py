import collections
import contextlib
import doctest
import functools
import importlib
import inspect
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union
from unittest import mock
from unittest.mock import patch
import urllib3
from transformers import logging as transformers_logging
from .integrations import (
from .integrations.deepspeed import is_deepspeed_available
from .utils import (
import asyncio  # noqa
class HfDoctestModule(Module):
    """
    Overwrites the `DoctestModule` of the pytest package to make sure the HFDocTestParser is used when discovering
    tests.
    """

    def collect(self) -> Iterable[DoctestItem]:

        class MockAwareDocTestFinder(doctest.DocTestFinder):
            """A hackish doctest finder that overrides stdlib internals to fix a stdlib bug.

            https://github.com/pytest-dev/pytest/issues/3456 https://bugs.python.org/issue25532
            """

            def _find_lineno(self, obj, source_lines):
                """Doctest code does not take into account `@property`, this
                is a hackish way to fix it. https://bugs.python.org/issue17446

                Wrapped Doctests will need to be unwrapped so the correct line number is returned. This will be
                reported upstream. #8796
                """
                if isinstance(obj, property):
                    obj = getattr(obj, 'fget', obj)
                if hasattr(obj, '__wrapped__'):
                    obj = inspect.unwrap(obj)
                return super()._find_lineno(obj, source_lines)

            def _find(self, tests, obj, name, module, source_lines, globs, seen) -> None:
                if _is_mocked(obj):
                    return
                with _patch_unwrap_mock_aware():
                    super()._find(tests, obj, name, module, source_lines, globs, seen)
        if self.path.name == 'conftest.py':
            module = self.config.pluginmanager._importconftest(self.path, self.config.getoption('importmode'), rootpath=self.config.rootpath)
        else:
            try:
                module = import_path(self.path, root=self.config.rootpath, mode=self.config.getoption('importmode'))
            except ImportError:
                if self.config.getvalue('doctest_ignore_import_errors'):
                    skip('unable to import module %r' % self.path)
                else:
                    raise
        finder = MockAwareDocTestFinder(parser=HfDocTestParser())
        optionflags = get_optionflags(self)
        runner = _get_runner(verbose=False, optionflags=optionflags, checker=_get_checker(), continue_on_failure=_get_continue_on_failure(self.config))
        for test in finder.find(module, module.__name__):
            if test.examples:
                yield DoctestItem.from_parent(self, name=test.name, runner=runner, dtest=test)