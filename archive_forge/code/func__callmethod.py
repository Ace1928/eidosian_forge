import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid
from traceback import format_exc
from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
def _callmethod(self, methodname, args=(), kwds={}):
    """
        Try to call a method of the referent and return a copy of the result
        """
    try:
        conn = self._tls.connection
    except AttributeError:
        util.debug('thread %r does not own a connection', threading.current_thread().name)
        self._connect()
        conn = self._tls.connection
    conn.send((self._id, methodname, args, kwds))
    kind, result = conn.recv()
    if kind == '#RETURN':
        return result
    elif kind == '#PROXY':
        exposed, token = result
        proxytype = self._manager._registry[token.typeid][-1]
        token.address = self._token.address
        proxy = proxytype(token, self._serializer, manager=self._manager, authkey=self._authkey, exposed=exposed)
        conn = self._Client(token.address, authkey=self._authkey)
        dispatch(conn, None, 'decref', (token.id,))
        return proxy
    raise convert_to_error(kind, result)