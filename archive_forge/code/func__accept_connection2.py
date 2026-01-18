import collections
import errno
import functools
import socket
import sys
import warnings
from . import base_events
from . import constants
from . import events
from . import futures
from . import selectors
from . import transports
from . import sslproto
from .coroutines import coroutine
from .log import logger
@coroutine
def _accept_connection2(self, protocol_factory, conn, extra, sslcontext=None, server=None):
    protocol = None
    transport = None
    try:
        protocol = protocol_factory()
        waiter = futures.Future(loop=self)
        if sslcontext:
            transport = self._make_ssl_transport(conn, protocol, sslcontext, waiter=waiter, server_side=True, extra=extra, server=server)
        else:
            transport = self._make_socket_transport(conn, protocol, waiter=waiter, extra=extra, server=server)
        try:
            yield from waiter
        except:
            transport.close()
            raise
    except Exception as exc:
        if self._debug:
            context = {'message': 'Error on transport creation for incoming connection', 'exception': exc}
            if protocol is not None:
                context['protocol'] = protocol
            if transport is not None:
                context['transport'] = transport
            self.call_exception_handler(context)