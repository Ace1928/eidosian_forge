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
def _get_save_args(state: State, filename: PathLike | None, resources: ResourcesLike | None, title: str | None) -> tuple[PathLike, Resources, str]:
    """

    """
    filename, is_default_filename = _get_save_filename(state, filename)
    resources = _get_save_resources(state, resources, is_default_filename)
    title = _get_save_title(state, title, is_default_filename)
    return (filename, resources, title)