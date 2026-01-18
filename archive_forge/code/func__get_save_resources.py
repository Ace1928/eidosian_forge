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
def _get_save_resources(state: State, resources: ResourcesLike | None, suppress_warning: bool) -> Resources:
    if resources is not None:
        if isinstance(resources, Resources):
            return resources
        else:
            return Resources(mode=resources)
    if state.file:
        return state.file.resources
    if not suppress_warning:
        warn('save() called but no resources were supplied and output_file(...) was never called, defaulting to resources.CDN')
    return Resources(mode=settings.resources())