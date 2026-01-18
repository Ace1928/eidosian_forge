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
def _mark_plugins_for_rewrite(self, hook) -> None:
    """Given an importhook, mark for rewrite any top-level
        modules or packages in the distribution package for
        all pytest plugins."""
    self.pluginmanager.rewrite_hook = hook
    if os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD'):
        return
    package_files = (str(file) for dist in importlib.metadata.distributions() if any((ep.group == 'pytest11' for ep in dist.entry_points)) for file in dist.files or [])
    for name in _iter_rewritable_modules(package_files):
        hook.mark_rewrite(name)