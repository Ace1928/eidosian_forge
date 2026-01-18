import argparse
import functools
import sys
import types
from typing import Any
from typing import Callable
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import unittest
from _pytest import outcomes
from _pytest._code import ExceptionInfo
from _pytest.config import Config
from _pytest.config import ConftestImportFailure
from _pytest.config import hookimpl
from _pytest.config import PytestPluginManager
from _pytest.config.argparsing import Parser
from _pytest.config.exceptions import UsageError
from _pytest.nodes import Node
from _pytest.reports import BaseReport
@classmethod
def _get_pdb_wrapper_class(cls, pdb_cls, capman: Optional['CaptureManager']):
    import _pytest.config

    class PytestPdbWrapper(pdb_cls):
        _pytest_capman = capman
        _continued = False

        def do_debug(self, arg):
            cls._recursive_debug += 1
            ret = super().do_debug(arg)
            cls._recursive_debug -= 1
            return ret

        def do_continue(self, arg):
            ret = super().do_continue(arg)
            if cls._recursive_debug == 0:
                assert cls._config is not None
                tw = _pytest.config.create_terminal_writer(cls._config)
                tw.line()
                capman = self._pytest_capman
                capturing = pytestPDB._is_capturing(capman)
                if capturing:
                    if capturing == 'global':
                        tw.sep('>', 'PDB continue (IO-capturing resumed)')
                    else:
                        tw.sep('>', 'PDB continue (IO-capturing resumed for %s)' % capturing)
                    assert capman is not None
                    capman.resume()
                else:
                    tw.sep('>', 'PDB continue')
            assert cls._pluginmanager is not None
            cls._pluginmanager.hook.pytest_leave_pdb(config=cls._config, pdb=self)
            self._continued = True
            return ret
        do_c = do_cont = do_continue

        def do_quit(self, arg):
            """Raise Exit outcome when quit command is used in pdb.

                This is a bit of a hack - it would be better if BdbQuit
                could be handled, but this would require to wrap the
                whole pytest run, and adjust the report etc.
                """
            ret = super().do_quit(arg)
            if cls._recursive_debug == 0:
                outcomes.exit('Quitting debugger')
            return ret
        do_q = do_quit
        do_exit = do_quit

        def setup(self, f, tb):
            """Suspend on setup().

                Needed after do_continue resumed, and entering another
                breakpoint again.
                """
            ret = super().setup(f, tb)
            if not ret and self._continued:
                if self._pytest_capman:
                    self._pytest_capman.suspend_global_capture(in_=True)
            return ret

        def get_stack(self, f, t):
            stack, i = super().get_stack(f, t)
            if f is None:
                i = max(0, len(stack) - 1)
                while i and stack[i][0].f_locals.get('__tracebackhide__', False):
                    i -= 1
            return (stack, i)
    return PytestPdbWrapper