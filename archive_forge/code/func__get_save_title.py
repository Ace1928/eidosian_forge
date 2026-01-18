from __future__ import annotations
import logging # isort:skip
from os.path import abspath, expanduser
from typing import Sequence
from jinja2 import Template
from ..core.templates import FILE
from ..core.types import PathLike
from ..models.ui import UIElement
from ..resources import Resources, ResourcesLike
from ..settings import settings
from ..themes import Theme
from ..util.warnings import warn
from .state import State, curstate
from .util import default_filename
def _get_save_title(state: State, title: str | None, suppress_warning: bool) -> str:
    if title is not None:
        return title
    if state.file:
        return state.file.title
    if not suppress_warning:
        warn("save() called but no title was supplied and output_file(...) was never called, using default title 'Bokeh Plot'")
    return DEFAULT_TITLE