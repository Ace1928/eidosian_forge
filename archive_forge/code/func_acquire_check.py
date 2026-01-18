import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def acquire_check(me, reader):
    if reader:
        lock_func = lock.read_lock
    else:
        lock_func = lock.write_lock
    with lock_func():
        if not reader:
            if len(active) >= 1:
                dups.append(me)
                dups.extend(active)
        active.append(me)
        try:
            time.sleep(random.random() / 100)
        finally:
            active.remove(me)