from __future__ import annotations
import contextlib
import datetime
import errno
import hashlib
import importlib
import importlib.util
import inspect
import locale
import os
import os.path
import re
import sys
import types
from types import ModuleType
from typing import (
from coverage import env
from coverage.exceptions import CoverageException
from coverage.types import TArc
from coverage.exceptions import *   # pylint: disable=wildcard-import
def _needs_to_implement(that: Any, func_name: str) -> NoReturn:
    """Helper to raise NotImplementedError in interface stubs."""
    if hasattr(that, '_coverage_plugin_name'):
        thing = 'Plugin'
        name = that._coverage_plugin_name
    else:
        thing = 'Class'
        klass = that.__class__
        name = f'{klass.__module__}.{klass.__name__}'
    raise NotImplementedError(f'{thing} {name!r} needs to implement {func_name}()')