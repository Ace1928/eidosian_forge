from __future__ import annotations
import logging # isort:skip
import pathlib
from typing import TYPE_CHECKING, Any as any
from ..core.has_props import HasProps, abstract
from ..core.properties import (
from ..core.property.bases import Init
from ..core.property.singletons import Intrinsic
from ..core.validation import error
from ..core.validation.errors import INVALID_PROPERTY_VALUE, NOT_A_PROPERTY_OF
from ..model import Model
class OpenURL(Callback):
    """ Open a URL in a new or current tab or window.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    url = String('http://', help='\n    The URL to direct the web browser to. This can be a template string,\n    which will be formatted with data from the data source.\n    ')
    same_tab = Bool(False, help='\n    Open URL in a new (`False`, default) or current (`True`) tab or window.\n    For `same_tab=False`, whether tab or window will be opened is browser\n    dependent.\n    ')