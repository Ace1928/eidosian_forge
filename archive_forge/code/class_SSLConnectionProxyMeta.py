import socket
import sys
import threading
import time
from . import Adapter
from .. import errors, server as cheroot_server
from ..makefile import StreamReader, StreamWriter
class SSLConnectionProxyMeta:
    """Metaclass for generating a bunch of proxy methods."""

    def __new__(mcl, name, bases, nmspc):
        """Attach a list of proxy methods to a new class."""
        proxy_methods = ('get_context', 'pending', 'send', 'write', 'recv', 'read', 'renegotiate', 'bind', 'listen', 'connect', 'accept', 'setblocking', 'fileno', 'close', 'get_cipher_list', 'getpeername', 'getsockname', 'getsockopt', 'setsockopt', 'makefile', 'get_app_data', 'set_app_data', 'state_string', 'sock_shutdown', 'get_peer_certificate', 'want_read', 'want_write', 'set_connect_state', 'set_accept_state', 'connect_ex', 'sendall', 'settimeout', 'gettimeout', 'shutdown')
        proxy_methods_no_args = ('shutdown',)
        proxy_props = ('family',)

        def lock_decorator(method):
            """Create a proxy method for a new class."""

            def proxy_wrapper(self, *args):
                self._lock.acquire()
                try:
                    new_args = args[:] if method not in proxy_methods_no_args else []
                    return getattr(self._ssl_conn, method)(*new_args)
                finally:
                    self._lock.release()
            return proxy_wrapper
        for m in proxy_methods:
            nmspc[m] = lock_decorator(m)
            nmspc[m].__name__ = m

        def make_property(property_):
            """Create a proxy method for a new class."""

            def proxy_prop_wrapper(self):
                return getattr(self._ssl_conn, property_)
            proxy_prop_wrapper.__name__ = property_
            return property(proxy_prop_wrapper)
        for p in proxy_props:
            nmspc[p] = make_property(p)
        return type(name, bases, nmspc)