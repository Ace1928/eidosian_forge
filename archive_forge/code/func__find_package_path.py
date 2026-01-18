from __future__ import annotations
import importlib.util
import os
import pathlib
import sys
import typing as t
from collections import defaultdict
from functools import update_wrapper
from jinja2 import BaseLoader
from jinja2 import FileSystemLoader
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException
from werkzeug.utils import cached_property
from .. import typing as ft
from ..helpers import get_root_path
from ..templating import _default_template_ctx_processor
def _find_package_path(import_name: str) -> str:
    """Find the path that contains the package or module."""
    root_mod_name, _, _ = import_name.partition('.')
    try:
        root_spec = importlib.util.find_spec(root_mod_name)
        if root_spec is None:
            raise ValueError('not found')
    except (ImportError, ValueError):
        return os.getcwd()
    if root_spec.submodule_search_locations:
        if root_spec.origin is None or root_spec.origin == 'namespace':
            package_spec = importlib.util.find_spec(import_name)
            if package_spec is not None and package_spec.submodule_search_locations:
                package_path = pathlib.Path(os.path.commonpath(package_spec.submodule_search_locations))
                search_location = next((location for location in root_spec.submodule_search_locations if _path_is_relative_to(package_path, location)))
            else:
                search_location = root_spec.submodule_search_locations[0]
            return os.path.dirname(search_location)
        else:
            return os.path.dirname(os.path.dirname(root_spec.origin))
    else:
        return os.path.dirname(root_spec.origin)