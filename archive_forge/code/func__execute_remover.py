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
def _execute_remover(self, callback_id: ID, removers: Removers) -> None:
    try:
        with self._removers_lock:
            remover = removers.pop(callback_id)
            for cb, cb_ids in list(self._get_removers_ids_by_callable(removers).items()):
                try:
                    cb_ids.remove(callback_id)
                    if not cb_ids:
                        del self._get_removers_ids_by_callable(removers)[cb]
                except KeyError:
                    pass
    except KeyError:
        raise ValueError("Removing a callback twice (or after it's already been run)")
    remover()