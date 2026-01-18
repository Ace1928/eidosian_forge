import asyncio
import inspect
import logging
import time
from functools import partial
import param
from ..util import edit_readonly, function_name
from .logging import LOG_PERIODIC_END, LOG_PERIODIC_START
from .state import curdoc_locked, set_curdoc, state
def _exec_callback(self, post=False):
    try:
        with set_curdoc(self._doc):
            if self.running:
                self.counter += 1
                if self.count is not None and self.counter > self.count:
                    self.stop()
            cb = self.callback() if self.running else None
    except Exception:
        cb = None
    if post:
        self._post_callback()
    return cb