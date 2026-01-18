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
def _link_props(self, model: Model, properties: List[str] | List[Tuple[str, str]], doc: Document, root: Model, comm: Optional[Comm]=None) -> None:
    from .config import config
    ref = root.ref['id']
    if config.embed:
        return
    for p in properties:
        if isinstance(p, tuple):
            _, p = p
        m = model
        if '.' in p:
            *subpath, p = p.split('.')
            for sp in subpath:
                m = getattr(m, sp)
        else:
            subpath = None
        if comm:
            m.on_change(p, partial(self._comm_change, doc, ref, comm, subpath))
        else:
            m.on_change(p, partial(self._server_change, doc, ref, subpath))