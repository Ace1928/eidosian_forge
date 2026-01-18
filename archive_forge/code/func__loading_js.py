from __future__ import annotations
import logging # isort:skip
import json
import os
import urllib
from typing import (
from uuid import uuid4
from ..core.types import ID
from ..util.serialization import make_id
from ..util.warnings import warn
from .state import curstate
def _loading_js(bundle: Bundle, element_id: ID | None, load_timeout: int=5000, register_mime: bool=True) -> str:
    """

    """
    from ..core.templates import AUTOLOAD_NB_JS
    return AUTOLOAD_NB_JS.render(bundle=bundle, elementid=element_id, force=True, timeout=load_timeout, register_mime=register_mime)