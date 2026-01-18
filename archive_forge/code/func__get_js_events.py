from __future__ import annotations
import json
import sys
from collections import defaultdict
from typing import (
import param
from bokeh.models import CustomJS
from pyviz_comms import JupyterComm
from ..util import lazy_load
from ..viewable import Viewable
from .base import ModelPane
def _get_js_events(self, ref):
    js_events = defaultdict(list)
    for event, specs in self._js_callbacks.items():
        for query, code, args in specs:
            models = {name: viewable._models[ref][0] for name, viewable in args.items() if ref in viewable._models}
            js_events[event].append({'query': query, 'callback': CustomJS(code=code, args=models)})
    return dict(js_events)