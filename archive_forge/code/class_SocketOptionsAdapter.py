import socket
import warnings
import sys
import requests
from requests import adapters
from .._compat import connection
from .._compat import poolmanager
from .. import exceptions as exc
class SocketOptionsAdapter(adapters.HTTPAdapter):
    """An adapter for requests that allows users to specify socket options.

    Since version 2.4.0 of requests, it is possible to specify a custom list
    of socket options that need to be set before establishing the connection.

    Example usage::

        >>> import socket
        >>> import requests
        >>> from requests_toolbelt.adapters import socket_options
        >>> s = requests.Session()
        >>> opts = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 0)]
        >>> adapter = socket_options.SocketOptionsAdapter(socket_options=opts)
        >>> s.mount('http://', adapter)

    You can also take advantage of the list of default options on this class
    to keep using the original options in addition to your custom options. In
    that case, ``opts`` might look like::

        >>> opts = socket_options.SocketOptionsAdapter.default_options + opts

    """
    if connection is not None:
        default_options = getattr(connection.HTTPConnection, 'default_socket_options', [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)])
    else:
        default_options = []
        warnings.warn(exc.RequestsVersionTooOld, 'This version of Requests is only compatible with a version of urllib3 which is too old to support setting options on a socket. This adapter is functionally useless.')

    def __init__(self, **kwargs):
        self.socket_options = kwargs.pop('socket_options', self.default_options)
        super(SocketOptionsAdapter, self).__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        if requests.__build__ >= 132096:
            self.poolmanager = poolmanager.PoolManager(num_pools=connections, maxsize=maxsize, block=block, socket_options=self.socket_options)
        else:
            super(SocketOptionsAdapter, self).init_poolmanager(connections, maxsize, block)