import sys
import os
import threading
import collections
import time
import types
import weakref
import errno
from queue import Empty, Full
from . import connection
from . import context
from .util import debug, info, Finalize, register_after_fork, is_exiting
@staticmethod
def _feed(buffer, notempty, send_bytes, writelock, reader_close, writer_close, ignore_epipe, onerror, queue_sem):
    debug('starting thread to feed data to pipe')
    nacquire = notempty.acquire
    nrelease = notempty.release
    nwait = notempty.wait
    bpopleft = buffer.popleft
    sentinel = _sentinel
    if sys.platform != 'win32':
        wacquire = writelock.acquire
        wrelease = writelock.release
    else:
        wacquire = None
    while 1:
        try:
            nacquire()
            try:
                if not buffer:
                    nwait()
            finally:
                nrelease()
            try:
                while 1:
                    obj = bpopleft()
                    if obj is sentinel:
                        debug('feeder thread got sentinel -- exiting')
                        reader_close()
                        writer_close()
                        return
                    obj = _ForkingPickler.dumps(obj)
                    if wacquire is None:
                        send_bytes(obj)
                    else:
                        wacquire()
                        try:
                            send_bytes(obj)
                        finally:
                            wrelease()
            except IndexError:
                pass
        except Exception as e:
            if ignore_epipe and getattr(e, 'errno', 0) == errno.EPIPE:
                return
            if is_exiting():
                info('error in queue thread: %s', e)
                return
            else:
                queue_sem.release()
                onerror(e, obj)