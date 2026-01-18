from __future__ import annotations
import logging # isort:skip
import hashlib
import json
import os
from dataclasses import dataclass
from os.path import normpath
from pathlib import Path
from typing import (
from urllib.parse import urljoin
from ..core.has_props import HasProps
from ..core.templates import CSS_RESOURCES, JS_RESOURCES
from ..document.document import Document
from ..resources import Resources
from ..settings import settings
from ..util.compiler import bundle_models
from .util import contains_tex_string
def _query_extensions(all_objs: set[HasProps], query: Callable[[type[HasProps]], bool]) -> bool:
    names: set[str] = set()
    for obj in all_objs:
        if hasattr(obj, '__implementation__'):
            continue
        name = obj.__view_module__.split('.')[0]
        if name == 'bokeh':
            continue
        if name in names:
            continue
        names.add(name)
        for model in HasProps.model_class_reverse_map.values():
            if model.__module__.startswith(name):
                if query(model):
                    return True
    return False