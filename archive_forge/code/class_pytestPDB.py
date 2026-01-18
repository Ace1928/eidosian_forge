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
class pytestPDB:
    """Pseudo PDB that defers to the real pdb."""
    _pluginmanager: Optional[PytestPluginManager] = None
    _config: Optional[Config] = None
    _saved: List[Tuple[Callable[..., None], Optional[PytestPluginManager], Optional[Config]]] = []
    _recursive_debug = 0
    _wrapped_pdb_cls: Optional[Tuple[Type[Any], Type[Any]]] = None

    @classmethod
    def _is_capturing(cls, capman: Optional['CaptureManager']) -> Union[str, bool]:
        if capman:
            return capman.is_capturing()
        return False

    @classmethod
    def _import_pdb_cls(cls, capman: Optional['CaptureManager']):
        if not cls._config:
            import pdb
            return pdb.Pdb
        usepdb_cls = cls._config.getvalue('usepdb_cls')
        if cls._wrapped_pdb_cls and cls._wrapped_pdb_cls[0] == usepdb_cls:
            return cls._wrapped_pdb_cls[1]
        if usepdb_cls:
            modname, classname = usepdb_cls
            try:
                __import__(modname)
                mod = sys.modules[modname]
                parts = classname.split('.')
                pdb_cls = getattr(mod, parts[0])
                for part in parts[1:]:
                    pdb_cls = getattr(pdb_cls, part)
            except Exception as exc:
                value = ':'.join((modname, classname))
                raise UsageError(f'--pdbcls: could not import {value!r}: {exc}') from exc
        else:
            import pdb
            pdb_cls = pdb.Pdb
        wrapped_cls = cls._get_pdb_wrapper_class(pdb_cls, capman)
        cls._wrapped_pdb_cls = (usepdb_cls, wrapped_cls)
        return wrapped_cls

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

    @classmethod
    def _init_pdb(cls, method, *args, **kwargs):
        """Initialize PDB debugging, dropping any IO capturing."""
        import _pytest.config
        if cls._pluginmanager is None:
            capman: Optional[CaptureManager] = None
        else:
            capman = cls._pluginmanager.getplugin('capturemanager')
        if capman:
            capman.suspend(in_=True)
        if cls._config:
            tw = _pytest.config.create_terminal_writer(cls._config)
            tw.line()
            if cls._recursive_debug == 0:
                header = kwargs.pop('header', None)
                if header is not None:
                    tw.sep('>', header)
                else:
                    capturing = cls._is_capturing(capman)
                    if capturing == 'global':
                        tw.sep('>', f'PDB {method} (IO-capturing turned off)')
                    elif capturing:
                        tw.sep('>', f'PDB {method} (IO-capturing turned off for {capturing})')
                    else:
                        tw.sep('>', f'PDB {method}')
        _pdb = cls._import_pdb_cls(capman)(**kwargs)
        if cls._pluginmanager:
            cls._pluginmanager.hook.pytest_enter_pdb(config=cls._config, pdb=_pdb)
        return _pdb

    @classmethod
    def set_trace(cls, *args, **kwargs) -> None:
        """Invoke debugging via ``Pdb.set_trace``, dropping any IO capturing."""
        frame = sys._getframe().f_back
        _pdb = cls._init_pdb('set_trace', *args, **kwargs)
        _pdb.set_trace(frame)