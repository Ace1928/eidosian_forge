import asyncio
import datetime
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import _api, backend_bases, backend_tools
from matplotlib.backends import backend_agg
from matplotlib.backend_bases import (
class TimerAsyncio(backend_bases.TimerBase):

    def __init__(self, *args, **kwargs):
        self._task = None
        super().__init__(*args, **kwargs)

    async def _timer_task(self, interval):
        while True:
            try:
                await asyncio.sleep(interval)
                self._on_timer()
                if self._single:
                    break
            except asyncio.CancelledError:
                break

    def _timer_start(self):
        self._timer_stop()
        self._task = asyncio.ensure_future(self._timer_task(max(self.interval / 1000.0, 1e-06)))

    def _timer_stop(self):
        if self._task is not None:
            self._task.cancel()
        self._task = None

    def _timer_set_interval(self):
        if self._task is not None:
            self._timer_stop()
            self._timer_start()