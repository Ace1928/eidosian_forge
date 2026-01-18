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
def _make_process_result(self, hub):
    """Coroutine reading messages from the pool processes."""
    fileno_to_outq = self.fileno_to_outq
    on_state_change = self.on_state_change
    add_reader = hub.add_reader
    remove_reader = hub.remove_reader
    recv_message = self._recv_message

    def on_result_readable(fileno):
        try:
            fileno_to_outq[fileno]
        except KeyError:
            return remove_reader(fileno)
        it = recv_message(add_reader, fileno, on_state_change)
        try:
            next(it)
        except StopIteration:
            pass
        except (OSError, EOFError):
            remove_reader(fileno)
        else:
            add_reader(fileno, it)
    return on_result_readable