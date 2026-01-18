from __future__ import annotations
import asyncio
import math
import os
import sys
import time
from math import pi
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource, FixedTicker, Tooltip
from bokeh.plotting import figure
from tqdm.asyncio import tqdm as _tqdm
from .._param import Align
from ..io.resources import CDN_DIST
from ..layout import Column, Panel, Row
from ..models import (
from ..pane.markup import Str
from ..reactive import SyncableData
from ..util import PARAM_NAME_PATTERN, escape, updating
from ..viewable import Viewable
from .base import Widget
class BooleanIndicator(Indicator):
    """
    BooleanIndicator is an abstract baseclass for indicators that
    visually indicate a boolean value.
    """
    throttle = param.Integer(default=500, doc='"\n        Throttles value change events, ensuring that they only toggle\n        off after a minimum time specified in milliseconds has passed.')
    value = param.Boolean(default=False, doc='\n        Whether the indicator is active or not.')
    _rename: ClassVar[Mapping[str, str | None]] = {'throttle': None}
    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        self._last__updated = time.monotonic()
        self._reset__task = None

    def _throttle_events(self, events):
        try:
            io_loop = asyncio.get_running_loop()
        except RuntimeError:
            return events
        if not io_loop.is_running():
            return events
        throttled_events = {}

        async def schedule_off():
            await asyncio.sleep(self.throttle / 1000)
            if self._reset__task:
                self.param.trigger('value')
            self._reset__task = None
        for k, e in events.items():
            if e.name != 'value':
                throttled_events[k] = e
                continue
            new_time = time.monotonic()
            if new_time - self._last__updated < self.throttle / 1000 and (not e.new):
                if self._reset__task:
                    self._reset__task.cancel()
                self._reset__task = asyncio.create_task(schedule_off())
                continue
            elif self._reset__task and e.new:
                self._last__updated = new_time
                self._reset__task.cancel()
                self._reset__task = None
                continue
            throttled_events[k] = e
            self._last__updated = new_time
        return throttled_events

    def _update_model(self, events: Dict[str, param.parameterized.Event], msg: Dict[str, Any], root: Model, model: Model, doc: Document, comm: Optional[Comm]) -> None:
        events = self._throttle_events(events)
        if not events:
            return
        super()._update_model(events, msg, root, model, doc, comm)