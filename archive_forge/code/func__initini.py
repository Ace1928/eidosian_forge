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
def _initini(self, args: Sequence[str]) -> None:
    ns, unknown_args = self._parser.parse_known_and_unknown_args(args, namespace=copy.copy(self.option))
    rootpath, inipath, inicfg = determine_setup(inifile=ns.inifilename, args=ns.file_or_dir + unknown_args, rootdir_cmd_arg=ns.rootdir or None, invocation_dir=self.invocation_params.dir)
    self._rootpath = rootpath
    self._inipath = inipath
    self.inicfg = inicfg
    self._parser.extra_info['rootdir'] = str(self.rootpath)
    self._parser.extra_info['inifile'] = str(self.inipath)
    self._parser.addini('addopts', 'Extra command line options', 'args')
    self._parser.addini('minversion', 'Minimally required pytest version')
    self._parser.addini('required_plugins', 'Plugins that must be present for pytest to run', type='args', default=[])
    self._override_ini = ns.override_ini or ()