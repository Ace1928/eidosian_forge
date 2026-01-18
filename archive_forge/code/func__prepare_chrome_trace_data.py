import os
import json
import atexit
import abc
import enum
import time
import threading
from timeit import default_timer as timer
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from numba.core import config
def _prepare_chrome_trace_data(listener: RecordingListener):
    """Prepare events in `listener` for serializing as chrome trace data.
    """
    pid = os.getpid()
    tid = threading.get_native_id()
    evs = []
    for ts, rec in listener.buffer:
        data = rec.data
        cat = str(rec.kind)
        ts_scaled = ts * 1000000
        ph = 'B' if rec.is_start else 'E'
        name = data['name']
        args = data
        ev = dict(cat=cat, pid=pid, tid=tid, ts=ts_scaled, ph=ph, name=name, args=args)
        evs.append(ev)
    return evs