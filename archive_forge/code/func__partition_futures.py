import collections
import contextlib
import functools
from concurrent import futures
from concurrent.futures import _base
import futurist
from futurist import _utils
def _partition_futures(fs):
    done = set()
    not_done = set()
    for f in fs:
        if f._state in _DONE_STATES:
            done.add(f)
        else:
            not_done.add(f)
    return (done, not_done)