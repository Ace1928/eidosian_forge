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
def _comm_event(self, doc: Document, event: Event) -> None:
    if state._thread_pool:
        future = state._thread_pool.submit(self._process_bokeh_event, doc, event)
        future.add_done_callback(partial(state._handle_future_exception, doc=doc))
    else:
        try:
            self._process_bokeh_event(doc, event)
        except Exception as e:
            state._handle_exception(e)