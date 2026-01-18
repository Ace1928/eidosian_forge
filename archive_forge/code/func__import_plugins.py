from __future__ import annotations
import configparser
import importlib.metadata
import inspect
import itertools
import logging
import sys
from typing import Any
from typing import Generator
from typing import Iterable
from typing import NamedTuple
from flake8 import utils
from flake8.defaults import VALID_CODE_PREFIX
from flake8.exceptions import ExecutionError
from flake8.exceptions import FailedToLoadPlugin
def _import_plugins(plugins: list[Plugin], opts: PluginOptions) -> list[LoadedPlugin]:
    sys.path.extend(opts.local_plugin_paths)
    return [_load_plugin(p) for p in plugins]