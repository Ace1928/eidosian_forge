from functools import partial
import sys
from eventlet import hubs, greenthread
from eventlet.greenio import GreenSocket
import eventlet.wsgi
import greenlet
from gunicorn.workers.base_async import AsyncWorker
from gunicorn.sock import ssl_wrap_socket
def _eventlet_serve(sock, handle, concurrency):
    """
    Serve requests forever.

    This code is nearly identical to ``eventlet.convenience.serve`` except
    that it attempts to join the pool at the end, which allows for gunicorn
    graceful shutdowns.
    """
    pool = eventlet.greenpool.GreenPool(concurrency)
    server_gt = eventlet.greenthread.getcurrent()
    while True:
        try:
            conn, addr = sock.accept()
            gt = pool.spawn(handle, conn, addr)
            gt.link(_eventlet_stop, server_gt, conn)
            conn, addr, gt = (None, None, None)
        except eventlet.StopServe:
            sock.close()
            pool.waitall()
            return