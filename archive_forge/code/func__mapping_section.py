from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
def _mapping_section(mapping, name, details_func, max_items_collapse, expand_option_name, enabled=True) -> str:
    n_items = len(mapping)
    expanded = _get_boolean_with_default(expand_option_name, n_items < max_items_collapse)
    collapsed = not expanded
    return collapsible_section(name, details=details_func(mapping), n_items=n_items, enabled=enabled, collapsed=collapsed)