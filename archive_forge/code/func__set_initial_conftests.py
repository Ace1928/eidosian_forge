import argparse
import collections.abc
import copy
import dataclasses
import enum
from functools import lru_cache
import glob
import importlib.metadata
import inspect
import os
from pathlib import Path
import re
import shlex
import sys
from textwrap import dedent
import types
from types import FunctionType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import warnings
import pluggy
from pluggy import HookimplMarker
from pluggy import HookimplOpts
from pluggy import HookspecMarker
from pluggy import HookspecOpts
from pluggy import PluginManager
from .compat import PathAwareHookProxy
from .exceptions import PrintHelp as PrintHelp
from .exceptions import UsageError as UsageError
from .findpaths import determine_setup
import _pytest._code
from _pytest._code import ExceptionInfo
from _pytest._code import filter_traceback
from _pytest._io import TerminalWriter
import _pytest.deprecated
import _pytest.hookspec
from _pytest.outcomes import fail
from _pytest.outcomes import Skipped
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import import_path
from _pytest.pathlib import ImportMode
from _pytest.pathlib import resolve_package_path
from _pytest.pathlib import safe_exists
from _pytest.stash import Stash
from _pytest.warning_types import PytestConfigWarning
from _pytest.warning_types import warn_explicit_for
def _set_initial_conftests(self, args: Sequence[Union[str, Path]], pyargs: bool, noconftest: bool, rootpath: Path, confcutdir: Optional[Path], invocation_dir: Path, importmode: Union[ImportMode, str], *, consider_namespace_packages: bool) -> None:
    """Load initial conftest files given a preparsed "namespace".

        As conftest files may add their own command line options which have
        arguments ('--my-opt somepath') we might get some false positives.
        All builtin and 3rd party plugins will have been loaded, however, so
        common options will not confuse our logic here.
        """
    self._confcutdir = absolutepath(invocation_dir / confcutdir) if confcutdir else None
    self._noconftest = noconftest
    self._using_pyargs = pyargs
    foundanchor = False
    for intitial_path in args:
        path = str(intitial_path)
        i = path.find('::')
        if i != -1:
            path = path[:i]
        anchor = absolutepath(invocation_dir / path)
        if safe_exists(anchor):
            self._try_load_conftest(anchor, importmode, rootpath, consider_namespace_packages=consider_namespace_packages)
            foundanchor = True
    if not foundanchor:
        self._try_load_conftest(invocation_dir, importmode, rootpath, consider_namespace_packages=consider_namespace_packages)