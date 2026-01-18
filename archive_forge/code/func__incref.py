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
def _incref(self):
    if self._owned_by_manager:
        util.debug('owned_by_manager skipped INCREF of %r', self._token.id)
        return
    conn = self._Client(self._token.address, authkey=self._authkey)
    dispatch(conn, None, 'incref', (self._id,))
    util.debug('INCREF %r', self._token.id)
    self._idset.add(self._id)
    state = self._manager and self._manager._state
    self._close = util.Finalize(self, BaseProxy._decref, args=(self._token, self._authkey, state, self._tls, self._idset, self._Client), exitpriority=10)