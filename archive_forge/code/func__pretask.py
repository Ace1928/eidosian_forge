from __future__ import annotations
import contextlib
import sys
import threading
import time
from timeit import default_timer
from dask.callbacks import Callback
from dask.utils import _deprecated
def _pretask(self, key, dsk, state):
    self._state = state
    if self._file is not None:
        self._file.flush()