from __future__ import annotations
import logging # isort:skip
import sys
import threading
from collections import defaultdict
from traceback import format_exception
from typing import (
import tornado
from tornado import gen
from ..core.types import ID
def _get_removers_ids_by_callable(self, removers: Removers) -> RemoversByCallable:
    if removers is self._next_tick_callback_removers:
        return self._next_tick_removers_by_callable
    elif removers is self._timeout_callback_removers:
        return self._timeout_removers_by_callable
    elif removers is self._periodic_callback_removers:
        return self._periodic_removers_by_callable
    else:
        raise RuntimeError('Unhandled removers', removers)