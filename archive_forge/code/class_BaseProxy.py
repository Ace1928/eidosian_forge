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
class BaseProxy(object):
    """
    A base for proxies of shared objects
    """
    _address_to_local = {}
    _mutex = util.ForkAwareThreadLock()

    def __init__(self, token, serializer, manager=None, authkey=None, exposed=None, incref=True, manager_owned=False):
        with BaseProxy._mutex:
            tls_idset = BaseProxy._address_to_local.get(token.address, None)
            if tls_idset is None:
                tls_idset = (util.ForkAwareLocal(), ProcessLocalSet())
                BaseProxy._address_to_local[token.address] = tls_idset
        self._tls = tls_idset[0]
        self._idset = tls_idset[1]
        self._token = token
        self._id = self._token.id
        self._manager = manager
        self._serializer = serializer
        self._Client = listener_client[serializer][1]
        self._owned_by_manager = manager_owned
        if authkey is not None:
            self._authkey = process.AuthenticationString(authkey)
        elif self._manager is not None:
            self._authkey = self._manager._authkey
        else:
            self._authkey = process.current_process().authkey
        if incref:
            self._incref()
        util.register_after_fork(self, BaseProxy._after_fork)

    def _connect(self):
        util.debug('making connection to manager')
        name = process.current_process().name
        if threading.current_thread().name != 'MainThread':
            name += '|' + threading.current_thread().name
        conn = self._Client(self._token.address, authkey=self._authkey)
        dispatch(conn, None, 'accept_connection', (name,))
        self._tls.connection = conn

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

    def _getvalue(self):
        """
        Get a copy of the value of the referent
        """
        return self._callmethod('#GETVALUE')

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

    @staticmethod
    def _decref(token, authkey, state, tls, idset, _Client):
        idset.discard(token.id)
        if state is None or state.value == State.STARTED:
            try:
                util.debug('DECREF %r', token.id)
                conn = _Client(token.address, authkey=authkey)
                dispatch(conn, None, 'decref', (token.id,))
            except Exception as e:
                util.debug('... decref failed %s', e)
        else:
            util.debug('DECREF %r -- manager already shutdown', token.id)
        if not idset and hasattr(tls, 'connection'):
            util.debug('thread %r has no more proxies so closing conn', threading.current_thread().name)
            tls.connection.close()
            del tls.connection

    def _after_fork(self):
        self._manager = None
        try:
            self._incref()
        except Exception as e:
            util.info('incref failed: %s' % e)

    def __reduce__(self):
        kwds = {}
        if get_spawning_popen() is not None:
            kwds['authkey'] = self._authkey
        if getattr(self, '_isauto', False):
            kwds['exposed'] = self._exposed_
            return (RebuildProxy, (AutoProxy, self._token, self._serializer, kwds))
        else:
            return (RebuildProxy, (type(self), self._token, self._serializer, kwds))

    def __deepcopy__(self, memo):
        return self._getvalue()

    def __repr__(self):
        return '<%s object, typeid %r at %#x>' % (type(self).__name__, self._token.typeid, id(self))

    def __str__(self):
        """
        Return representation of the referent (or a fall-back if that fails)
        """
        try:
            return self._callmethod('__repr__')
        except Exception:
            return repr(self)[:-1] + "; '__str__()' failed>"