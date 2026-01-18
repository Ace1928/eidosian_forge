import errno
import gc
import inspect
import os
import select
import time
from collections import Counter, deque, namedtuple
from io import BytesIO
from numbers import Integral
from pickle import HIGHEST_PROTOCOL
from struct import pack, unpack, unpack_from
from time import sleep
from weakref import WeakValueDictionary, ref
from billiard import pool as _pool
from billiard.compat import isblocking, setblocking
from billiard.pool import ACK, NACK, RUN, TERMINATE, WorkersJoined
from billiard.queues import _SimpleQueue
from kombu.asynchronous import ERR, WRITE
from kombu.serialization import pickle as _pickle
from kombu.utils.eventio import SELECT_BAD_FD
from kombu.utils.functional import fxrange
from vine import promise
from celery.signals import worker_before_create_process
from celery.utils.functional import noop
from celery.utils.log import get_logger
from celery.worker import state as worker_state
def _write_job(proc, fd, job):
    header, body, body_size = job._payload
    errors = 0
    try:
        job._write_to = proc
        send = proc.send_job_offset
        Hw = Bw = 0
        while Hw < 4:
            try:
                Hw += send(header, Hw)
            except Exception as exc:
                if getattr(exc, 'errno', None) not in UNAVAIL:
                    raise
                errors += 1
                if errors > 100:
                    on_not_recovering(proc, fd, job, exc)
                    raise StopIteration()
                yield
            else:
                errors = 0
        while Bw < body_size:
            try:
                Bw += send(body, Bw)
            except Exception as exc:
                if getattr(exc, 'errno', None) not in UNAVAIL:
                    raise
                errors += 1
                if errors > 100:
                    on_not_recovering(proc, fd, job, exc)
                    raise StopIteration()
                yield
            else:
                errors = 0
    finally:
        hub_remove(fd)
        write_stats[proc.index] += 1
        active_writes.discard(fd)
        write_generator_done(job._writer())