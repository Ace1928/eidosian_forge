from __future__ import annotations
import datetime as dt
import difflib
import inspect
import logging
import re
import sys
import textwrap
from collections import Counter, defaultdict, namedtuple
from functools import lru_cache, partial
from pprint import pformat
from typing import (
import numpy as np
import param
from bokeh.core.property.descriptors import UnsetValueError
from bokeh.model import DataModel
from bokeh.models import ImportedStyleSheet
from packaging.version import Version
from param.parameterized import (
from .io.document import unlocked
from .io.model import hold
from .io.notebook import push
from .io.resources import (
from .io.state import set_curdoc, state
from .models.reactive_html import (
from .util import (
from .viewable import Layoutable, Renderable, Viewable
def _set_on_model(self, msg: Mapping[str, Any], root: Model, model: Model) -> None:
    if not msg:
        return
    ref = root.ref['id']
    old = self._changing.get(ref, [])
    self._changing[ref] = [attr for attr, value in msg.items() if not model.lookup(attr).property.matches(getattr(model, attr), value)]
    try:
        model.update(**msg)
    finally:
        if old:
            self._changing[ref] = old
        else:
            del self._changing[ref]
    if isinstance(model, DataModel):
        self._patch_datamodel_ref(model.properties_with_values(), ref)