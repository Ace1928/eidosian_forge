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
class BaseManager(object):
    """
    Base class for managers
    """
    _registry = {}
    _Server = Server

    def __init__(self, address=None, authkey=None, serializer='pickle', ctx=None):
        if authkey is None:
            authkey = process.current_process().authkey
        self._address = address
        self._authkey = process.AuthenticationString(authkey)
        self._state = State()
        self._state.value = State.INITIAL
        self._serializer = serializer
        self._Listener, self._Client = listener_client[serializer]
        self._ctx = ctx or get_context()

    def get_server(self):
        """
        Return server object with serve_forever() method and address attribute
        """
        if self._state.value != State.INITIAL:
            if self._state.value == State.STARTED:
                raise ProcessError('Already started server')
            elif self._state.value == State.SHUTDOWN:
                raise ProcessError('Manager has shut down')
            else:
                raise ProcessError('Unknown state {!r}'.format(self._state.value))
        return Server(self._registry, self._address, self._authkey, self._serializer)

    def connect(self):
        """
        Connect manager object to the server process
        """
        Listener, Client = listener_client[self._serializer]
        conn = Client(self._address, authkey=self._authkey)
        dispatch(conn, None, 'dummy')
        self._state.value = State.STARTED

    def start(self, initializer=None, initargs=()):
        """
        Spawn a server process for this manager object
        """
        if self._state.value != State.INITIAL:
            if self._state.value == State.STARTED:
                raise ProcessError('Already started server')
            elif self._state.value == State.SHUTDOWN:
                raise ProcessError('Manager has shut down')
            else:
                raise ProcessError('Unknown state {!r}'.format(self._state.value))
        if initializer is not None and (not callable(initializer)):
            raise TypeError('initializer must be a callable')
        reader, writer = connection.Pipe(duplex=False)
        self._process = self._ctx.Process(target=type(self)._run_server, args=(self._registry, self._address, self._authkey, self._serializer, writer, initializer, initargs))
        ident = ':'.join((str(i) for i in self._process._identity))
        self._process.name = type(self).__name__ + '-' + ident
        self._process.start()
        writer.close()
        self._address = reader.recv()
        reader.close()
        self._state.value = State.STARTED
        self.shutdown = util.Finalize(self, type(self)._finalize_manager, args=(self._process, self._address, self._authkey, self._state, self._Client), exitpriority=0)

    @classmethod
    def _run_server(cls, registry, address, authkey, serializer, writer, initializer=None, initargs=()):
        """
        Create a server, report its address and run it
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if initializer is not None:
            initializer(*initargs)
        server = cls._Server(registry, address, authkey, serializer)
        writer.send(server.address)
        writer.close()
        util.info('manager serving at %r', server.address)
        server.serve_forever()

    def _create(self, typeid, /, *args, **kwds):
        """
        Create a new shared object; return the token and exposed tuple
        """
        assert self._state.value == State.STARTED, 'server not yet started'
        conn = self._Client(self._address, authkey=self._authkey)
        try:
            id, exposed = dispatch(conn, None, 'create', (typeid,) + args, kwds)
        finally:
            conn.close()
        return (Token(typeid, self._address, id), exposed)

    def join(self, timeout=None):
        """
        Join the manager process (if it has been spawned)
        """
        if self._process is not None:
            self._process.join(timeout)
            if not self._process.is_alive():
                self._process = None

    def _debug_info(self):
        """
        Return some info about the servers shared objects and connections
        """
        conn = self._Client(self._address, authkey=self._authkey)
        try:
            return dispatch(conn, None, 'debug_info')
        finally:
            conn.close()

    def _number_of_objects(self):
        """
        Return the number of shared objects
        """
        conn = self._Client(self._address, authkey=self._authkey)
        try:
            return dispatch(conn, None, 'number_of_objects')
        finally:
            conn.close()

    def __enter__(self):
        if self._state.value == State.INITIAL:
            self.start()
        if self._state.value != State.STARTED:
            if self._state.value == State.INITIAL:
                raise ProcessError('Unable to start server')
            elif self._state.value == State.SHUTDOWN:
                raise ProcessError('Manager has shut down')
            else:
                raise ProcessError('Unknown state {!r}'.format(self._state.value))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    @staticmethod
    def _finalize_manager(process, address, authkey, state, _Client):
        """
        Shutdown the manager process; will be registered as a finalizer
        """
        if process.is_alive():
            util.info('sending shutdown message to manager')
            try:
                conn = _Client(address, authkey=authkey)
                try:
                    dispatch(conn, None, 'shutdown')
                finally:
                    conn.close()
            except Exception:
                pass
            process.join(timeout=1.0)
            if process.is_alive():
                util.info('manager still alive')
                if hasattr(process, 'terminate'):
                    util.info('trying to `terminate()` manager process')
                    process.terminate()
                    process.join(timeout=1.0)
                    if process.is_alive():
                        util.info('manager still alive after terminate')
        state.value = State.SHUTDOWN
        try:
            del BaseProxy._address_to_local[address]
        except KeyError:
            pass

    @property
    def address(self):
        return self._address

    @classmethod
    def register(cls, typeid, callable=None, proxytype=None, exposed=None, method_to_typeid=None, create_method=True):
        """
        Register a typeid with the manager type
        """
        if '_registry' not in cls.__dict__:
            cls._registry = cls._registry.copy()
        if proxytype is None:
            proxytype = AutoProxy
        exposed = exposed or getattr(proxytype, '_exposed_', None)
        method_to_typeid = method_to_typeid or getattr(proxytype, '_method_to_typeid_', None)
        if method_to_typeid:
            for key, value in list(method_to_typeid.items()):
                assert type(key) is str, '%r is not a string' % key
                assert type(value) is str, '%r is not a string' % value
        cls._registry[typeid] = (callable, exposed, method_to_typeid, proxytype)
        if create_method:

            def temp(self, /, *args, **kwds):
                util.debug('requesting creation of a shared %r object', typeid)
                token, exp = self._create(typeid, *args, **kwds)
                proxy = proxytype(token, self._serializer, manager=self, authkey=self._authkey, exposed=exp)
                conn = self._Client(token.address, authkey=self._authkey)
                dispatch(conn, None, 'decref', (token.id,))
                return proxy
            temp.__name__ = typeid
            setattr(cls, typeid, temp)