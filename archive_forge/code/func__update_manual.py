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
def _update_manual(self, *events: param.parameterized.Event) -> None:
    """
        Skip events triggered internally
        """
    processed_events = []
    for e in events:
        if e.name == self._data_params[0] and e.type == 'triggered' and self._updating:
            continue
        processed_events.append(e)
    super()._update_manual(*processed_events)