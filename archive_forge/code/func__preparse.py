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
def _preparse(self, args: List[str], addopts: bool=True) -> None:
    if addopts:
        env_addopts = os.environ.get('PYTEST_ADDOPTS', '')
        if len(env_addopts):
            args[:] = self._validate_args(shlex.split(env_addopts), 'via PYTEST_ADDOPTS') + args
    self._initini(args)
    if addopts:
        args[:] = self._validate_args(self.getini('addopts'), 'via addopts config') + args
    self.known_args_namespace = self._parser.parse_known_args(args, namespace=copy.copy(self.option))
    self._checkversion()
    self._consider_importhook(args)
    self.pluginmanager.consider_preparse(args, exclude_only=False)
    if not os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD'):
        self.pluginmanager.load_setuptools_entrypoints('pytest11')
    self.pluginmanager.consider_env()
    self.known_args_namespace = self._parser.parse_known_args(args, namespace=copy.copy(self.known_args_namespace))
    self._validate_plugins()
    self._warn_about_skipped_plugins()
    if self.known_args_namespace.confcutdir is None:
        if self.inipath is not None:
            confcutdir = str(self.inipath.parent)
        else:
            confcutdir = str(self.rootpath)
        self.known_args_namespace.confcutdir = confcutdir
    try:
        self.hook.pytest_load_initial_conftests(early_config=self, args=args, parser=self._parser)
    except ConftestImportFailure as e:
        if self.known_args_namespace.help or self.known_args_namespace.version:
            self.issue_config_time_warning(PytestConfigWarning(f'could not load initial conftests: {e.path}'), stacklevel=2)
        else:
            raise