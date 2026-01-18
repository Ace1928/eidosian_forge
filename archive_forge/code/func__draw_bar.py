from __future__ import annotations
import contextlib
import sys
import threading
import time
from timeit import default_timer
from dask.callbacks import Callback
from dask.utils import _deprecated
def _draw_bar(self, frac, elapsed):
    from dask.utils import format_time
    bar = '#' * int(self._width * frac)
    percent = int(100 * frac)
    elapsed = format_time(elapsed)
    msg = '\r[{0:<{1}}] | {2}% Completed | {3}'.format(bar, self._width, percent, elapsed)
    with contextlib.suppress(ValueError):
        if self._file is not None:
            self._file.write(msg)
            self._file.flush()