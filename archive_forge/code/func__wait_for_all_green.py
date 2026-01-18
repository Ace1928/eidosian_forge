import collections
import contextlib
import functools
from concurrent import futures
from concurrent.futures import _base
import futurist
from futurist import _utils
@_ensure_eventlet
def _wait_for_all_green(fs, timeout=None):
    if not fs:
        return DoneAndNotDoneFutures(set(), set())
    with _acquire_and_release_futures(fs):
        done, not_done = _partition_futures(fs)
        if len(done) == len(fs):
            return DoneAndNotDoneFutures(done, not_done)
        waiter = _create_and_install_waiters(not_done, _AllGreenWaiter, len(not_done))
    waiter.event.wait(timeout)
    for f in not_done:
        with f._condition:
            f._waiters.remove(waiter)
    with _acquire_and_release_futures(fs):
        done, not_done = _partition_futures(fs)
        return DoneAndNotDoneFutures(done, not_done)