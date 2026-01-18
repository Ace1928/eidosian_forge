from __future__ import annotations
import ast
import collections.abc as cabc
import importlib.metadata
import inspect
import os
import platform
import re
import sys
import traceback
import typing as t
from functools import update_wrapper
from operator import itemgetter
from types import ModuleType
import click
from click.core import ParameterSource
from werkzeug import run_simple
from werkzeug.serving import is_running_from_reloader
from werkzeug.utils import import_string
from .globals import current_app
from .helpers import get_debug_flag
from .helpers import get_load_dotenv
def _load_plugin_commands(self) -> None:
    if self._loaded_plugin_commands:
        return
    if sys.version_info >= (3, 10):
        from importlib import metadata
    else:
        import importlib_metadata as metadata
    for ep in metadata.entry_points(group='flask.commands'):
        self.add_command(ep.load(), ep.name)
    self._loaded_plugin_commands = True