from __future__ import annotations
import argparse
import enum
import functools
import logging
from typing import Any
from typing import Callable
from typing import Sequence
from flake8 import utils
from flake8.plugins.finder import Plugins
def _set_group(name: str) -> None:
    try:
        self._current_group = groups[name]
    except KeyError:
        group = self.parser.add_argument_group(name)
        self._current_group = groups[name] = group