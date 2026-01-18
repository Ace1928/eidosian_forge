import asyncio
import inspect
import logging
import time
from functools import partial
import param
from ..util import edit_readonly, function_name
from .logging import LOG_PERIODIC_END, LOG_PERIODIC_START
from .state import curdoc_locked, set_curdoc, state
@param.depends('period', watch=True)
def _update_period(self):
    if self._cb:
        self.stop()
        self.start()