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
def _getini(self, name: str):
    try:
        description, type, default = self._parser._inidict[name]
    except KeyError as e:
        raise ValueError(f'unknown configuration value: {name!r}') from e
    override_value = self._get_override_ini_value(name)
    if override_value is None:
        try:
            value = self.inicfg[name]
        except KeyError:
            return default
    else:
        value = override_value
    if type == 'paths':
        dp = self.inipath.parent if self.inipath is not None else self.invocation_params.dir
        input_values = shlex.split(value) if isinstance(value, str) else value
        return [dp / x for x in input_values]
    elif type == 'args':
        return shlex.split(value) if isinstance(value, str) else value
    elif type == 'linelist':
        if isinstance(value, str):
            return [t for t in map(lambda x: x.strip(), value.split('\n')) if t]
        else:
            return value
    elif type == 'bool':
        return _strtobool(str(value).strip())
    elif type == 'string':
        return value
    elif type is None:
        return value
    else:
        return self._getini_unknown_type(name, type, value)