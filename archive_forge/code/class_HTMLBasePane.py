from __future__ import annotations
import functools
import json
import textwrap
from typing import (
import param  # type: ignore
from ..io.resources import CDN_DIST
from ..models import HTML as _BkHTML, JSON as _BkJSON
from ..util import HTML_SANITIZER, escape
from .base import ModelPane
class HTMLBasePane(ModelPane):
    """
    Baseclass for Panes which render HTML inside a Bokeh Div.
    See the documentation for Bokeh Div for more detail about
    the supported options like style and sizing_mode.
    """
    _bokeh_model: ClassVar[Model] = _BkHTML
    _rename: ClassVar[Mapping[str, str | None]] = {'object': 'text'}
    _updates: ClassVar[bool] = True
    __abstract = True