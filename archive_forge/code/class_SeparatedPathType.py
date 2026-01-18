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
class SeparatedPathType(click.Path):
    """Click option type that accepts a list of values separated by the
    OS's path separator (``:``, ``;`` on Windows). Each value is
    validated as a :class:`click.Path` type.
    """

    def convert(self, value: t.Any, param: click.Parameter | None, ctx: click.Context | None) -> t.Any:
        items = self.split_envvar_value(value)
        super_convert = super().convert
        return [super_convert(item, param, ctx) for item in items]