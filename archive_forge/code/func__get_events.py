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
def _get_events(self) -> Dict[str, Dict[str, bool]]:
    events = {}
    for node, dom_events in self._dom_events.items():
        if isinstance(dom_events, list):
            events[node] = {e: True for e in dom_events}
        else:
            events[node] = dom_events
    for node, evs in self._event_callbacks.items():
        events[node] = node_events = events.get(node, {})
        for e in evs:
            if e not in node_events:
                node_events[e] = False
    return events