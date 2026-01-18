from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed"""
    return [files(package).joinpath(resource).read_text(encoding='utf-8') for package, resource in STATIC_FILES]