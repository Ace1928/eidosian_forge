import ast
from collections import defaultdict
import errno
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import io
import itertools
import marshal
import os
from pathlib import Path
from pathlib import PurePath
import struct
import sys
import tokenize
import types
from typing import Callable
from typing import Dict
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from _pytest._io.saferepr import DEFAULT_REPR_MAX_SIZE
from _pytest._io.saferepr import saferepr
from _pytest._version import version
from _pytest.assertion import util
from _pytest.config import Config
from _pytest.main import Session
from _pytest.pathlib import absolutepath
from _pytest.pathlib import fnmatch_ex
from _pytest.stash import StashKey
from _pytest.assertion.util import format_explanation as _format_explanation  # noqa:F401, isort:skip
class AssertionRewritingHook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """PEP302/PEP451 import hook which rewrites asserts."""

    def __init__(self, config: Config) -> None:
        self.config = config
        try:
            self.fnpats = config.getini('python_files')
        except ValueError:
            self.fnpats = ['test_*.py', '*_test.py']
        self.session: Optional[Session] = None
        self._rewritten_names: Dict[str, Path] = {}
        self._must_rewrite: Set[str] = set()
        self._writing_pyc = False
        self._basenames_to_check_rewrite = {'conftest'}
        self._marked_for_rewrite_cache: Dict[str, bool] = {}
        self._session_paths_checked = False

    def set_session(self, session: Optional[Session]) -> None:
        self.session = session
        self._session_paths_checked = False
    _find_spec = importlib.machinery.PathFinder.find_spec

    def find_spec(self, name: str, path: Optional[Sequence[Union[str, bytes]]]=None, target: Optional[types.ModuleType]=None) -> Optional[importlib.machinery.ModuleSpec]:
        if self._writing_pyc:
            return None
        state = self.config.stash[assertstate_key]
        if self._early_rewrite_bailout(name, state):
            return None
        state.trace('find_module called for: %s' % name)
        spec = self._find_spec(name, path)
        if spec is None or spec.origin is None or (not isinstance(spec.loader, importlib.machinery.SourceFileLoader)) or (not os.path.exists(spec.origin)):
            return None
        else:
            fn = spec.origin
        if not self._should_rewrite(name, fn, state):
            return None
        return importlib.util.spec_from_file_location(name, fn, loader=self, submodule_search_locations=spec.submodule_search_locations)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Optional[types.ModuleType]:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        assert module.__spec__ is not None
        assert module.__spec__.origin is not None
        fn = Path(module.__spec__.origin)
        state = self.config.stash[assertstate_key]
        self._rewritten_names[module.__name__] = fn
        write = not sys.dont_write_bytecode
        cache_dir = get_cache_dir(fn)
        if write:
            ok = try_makedirs(cache_dir)
            if not ok:
                write = False
                state.trace(f'read only directory: {cache_dir}')
        cache_name = fn.name[:-3] + PYC_TAIL
        pyc = cache_dir / cache_name
        co = _read_pyc(fn, pyc, state.trace)
        if co is None:
            state.trace(f'rewriting {fn!r}')
            source_stat, co = _rewrite_test(fn, self.config)
            if write:
                self._writing_pyc = True
                try:
                    _write_pyc(state, co, source_stat, pyc)
                finally:
                    self._writing_pyc = False
        else:
            state.trace(f'found cached rewritten pyc for {fn}')
        exec(co, module.__dict__)

    def _early_rewrite_bailout(self, name: str, state: 'AssertionState') -> bool:
        """A fast way to get out of rewriting modules.

        Profiling has shown that the call to PathFinder.find_spec (inside of
        the find_spec from this class) is a major slowdown, so, this method
        tries to filter what we're sure won't be rewritten before getting to
        it.
        """
        if self.session is not None and (not self._session_paths_checked):
            self._session_paths_checked = True
            for initial_path in self.session._initialpaths:
                parts = str(initial_path).split(os.sep)
                self._basenames_to_check_rewrite.add(os.path.splitext(parts[-1])[0])
        parts = name.split('.')
        if parts[-1] in self._basenames_to_check_rewrite:
            return False
        path = PurePath(*parts).with_suffix('.py')
        for pat in self.fnpats:
            if os.path.dirname(pat):
                return False
            if fnmatch_ex(pat, path):
                return False
        if self._is_marked_for_rewrite(name, state):
            return False
        state.trace(f'early skip of rewriting module: {name}')
        return True

    def _should_rewrite(self, name: str, fn: str, state: 'AssertionState') -> bool:
        if os.path.basename(fn) == 'conftest.py':
            state.trace(f'rewriting conftest file: {fn!r}')
            return True
        if self.session is not None:
            if self.session.isinitpath(absolutepath(fn)):
                state.trace(f'matched test file (was specified on cmdline): {fn!r}')
                return True
        fn_path = PurePath(fn)
        for pat in self.fnpats:
            if fnmatch_ex(pat, fn_path):
                state.trace(f'matched test file {fn!r}')
                return True
        return self._is_marked_for_rewrite(name, state)

    def _is_marked_for_rewrite(self, name: str, state: 'AssertionState') -> bool:
        try:
            return self._marked_for_rewrite_cache[name]
        except KeyError:
            for marked in self._must_rewrite:
                if name == marked or name.startswith(marked + '.'):
                    state.trace(f'matched marked file {name!r} (from {marked!r})')
                    self._marked_for_rewrite_cache[name] = True
                    return True
            self._marked_for_rewrite_cache[name] = False
            return False

    def mark_rewrite(self, *names: str) -> None:
        """Mark import names as needing to be rewritten.

        The named module or package as well as any nested modules will
        be rewritten on import.
        """
        already_imported = set(names).intersection(sys.modules).difference(self._rewritten_names)
        for name in already_imported:
            mod = sys.modules[name]
            if not AssertionRewriter.is_rewrite_disabled(mod.__doc__ or '') and (not isinstance(mod.__loader__, type(self))):
                self._warn_already_imported(name)
        self._must_rewrite.update(names)
        self._marked_for_rewrite_cache.clear()

    def _warn_already_imported(self, name: str) -> None:
        from _pytest.warning_types import PytestAssertRewriteWarning
        self.config.issue_config_time_warning(PytestAssertRewriteWarning('Module already imported so cannot be rewritten: %s' % name), stacklevel=5)

    def get_data(self, pathname: Union[str, bytes]) -> bytes:
        """Optional PEP302 get_data API."""
        with open(pathname, 'rb') as f:
            return f.read()
    if sys.version_info >= (3, 10):
        if sys.version_info >= (3, 12):
            from importlib.resources.abc import TraversableResources
        else:
            from importlib.abc import TraversableResources

        def get_resource_reader(self, name: str) -> TraversableResources:
            if sys.version_info < (3, 11):
                from importlib.readers import FileReader
            else:
                from importlib.resources.readers import FileReader
            return FileReader(types.SimpleNamespace(path=self._rewritten_names[name]))