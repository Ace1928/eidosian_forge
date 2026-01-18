from functools import partial
import sys
from eventlet import hubs, greenthread
from eventlet.greenio import GreenSocket
import eventlet.wsgi
import greenlet
from gunicorn.workers.base_async import AsyncWorker
from gunicorn.sock import ssl_wrap_socket
def _eventlet_stop(client, server, conn):
    """
    Stop a greenlet handling a request and close its connection.

    This code is lifted from eventlet so as not to depend on undocumented
    functions in the library.
    """
    try:
        try:
            client.wait()
        finally:
            conn.close()
    except greenlet.GreenletExit:
        pass
    except Exception:
        greenthread.kill(server, *sys.exc_info())