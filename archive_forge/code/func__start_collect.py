from __future__ import annotations
from collections import namedtuple
from itertools import starmap
from multiprocessing import Pipe, Process, current_process
from time import sleep
from timeit import default_timer
from dask.callbacks import Callback
from dask.utils import import_required
def _start_collect(self):
    if not self._is_running():
        self._tracker = _Tracker(self._dt)
        self._tracker.start()
    self._tracker.parent_conn.send('collect')