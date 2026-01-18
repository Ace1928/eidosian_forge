import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def _spawn_variation(readers, writers, max_workers=None):
    start_stops = collections.deque()
    lock = fasteners.ReaderWriterLock()

    def read_func(ident):
        with lock.read_lock():
            enter_time = _utils.now()
            time.sleep(WORK_TIMES[ident % len(WORK_TIMES)])
            exit_time = _utils.now()
            start_stops.append((lock.READER, enter_time, exit_time))
            time.sleep(NAPPY_TIME)

    def write_func(ident):
        with lock.write_lock():
            enter_time = _utils.now()
            time.sleep(WORK_TIMES[ident % len(WORK_TIMES)])
            exit_time = _utils.now()
            start_stops.append((lock.WRITER, enter_time, exit_time))
            time.sleep(NAPPY_TIME)
    if max_workers is None:
        max_workers = max(0, readers) + max(0, writers)
    if max_workers > 0:
        with futures.ThreadPoolExecutor(max_workers=max_workers) as e:
            count = 0
            for _i in range(0, readers):
                e.submit(read_func, count)
                count += 1
            for _i in range(0, writers):
                e.submit(write_func, count)
                count += 1
    writer_times = []
    reader_times = []
    for lock_type, start, stop in list(start_stops):
        if lock_type == lock.WRITER:
            writer_times.append((start, stop))
        else:
            reader_times.append((start, stop))
    return (writer_times, reader_times)