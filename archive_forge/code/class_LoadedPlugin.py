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
class LoadedPlugin(NamedTuple):
    """Represents a plugin after being imported."""
    plugin: Plugin
    obj: Any
    parameters: dict[str, bool]

    @property
    def entry_name(self) -> str:
        """Return the name given in the packaging metadata."""
        return self.plugin.entry_point.name

    @property
    def display_name(self) -> str:
        """Return the name for use in user-facing / error messages."""
        return f'{self.plugin.package}[{self.entry_name}]'