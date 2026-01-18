from __future__ import annotations
import ast
import json
from base64 import b64decode
from datetime import date, datetime
from typing import (
import numpy as np
import param
from bokeh.models.formatters import TickFormatter
from bokeh.models.widgets import (
from ..config import config
from ..layout import Column, Panel
from ..models import (
from ..util import param_reprs, try_datetime64_to_datetime
from .base import CompositeWidget, Widget
class _BooleanWidget(Widget):
    value = param.Boolean(default=False, doc='\n        The current value')
    _supports_embed: ClassVar[bool] = True
    _rename: ClassVar[Mapping[str, str | None]] = {'value': 'active', 'name': 'label'}
    __abstract = True

    def _get_embed_state(self, root, values=None, max_opts=3):
        return (self, self._models[root.ref['id']][0], [False, True], lambda x: x.active, 'active', 'cb_obj.active')