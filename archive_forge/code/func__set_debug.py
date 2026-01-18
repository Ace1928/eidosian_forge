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
def _set_debug(ctx: click.Context, param: click.Option, value: bool) -> bool | None:
    source = ctx.get_parameter_source(param.name)
    if source is not None and source in (ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP):
        return None
    os.environ['FLASK_DEBUG'] = '1' if value else '0'
    return value