from __future__ import annotations
import functools
import numbers
import socket
from bisect import bisect
from collections import namedtuple
from contextlib import contextmanager
from queue import Empty
from time import time
from vine import promise
from kombu.exceptions import InconsistencyError, VersionMismatch
from kombu.log import get_logger
from kombu.utils.compat import register_after_fork
from kombu.utils.encoding import bytes_to_str
from kombu.utils.eventio import ERR, READ, poll
from kombu.utils.functional import accepts_argument
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from kombu.utils.scheduling import cycle_by_name
from kombu.utils.url import _parse_url
from . import virtual
def _connparams(self, asynchronous=False):
    conninfo = self.connection.client
    connparams = {'host': conninfo.hostname or '127.0.0.1', 'port': conninfo.port or self.connection.default_port, 'virtual_host': conninfo.virtual_host, 'username': conninfo.userid, 'password': conninfo.password, 'max_connections': self.max_connections, 'socket_timeout': self.socket_timeout, 'socket_connect_timeout': self.socket_connect_timeout, 'socket_keepalive': self.socket_keepalive, 'socket_keepalive_options': self.socket_keepalive_options, 'health_check_interval': self.health_check_interval, 'retry_on_timeout': self.retry_on_timeout}
    conn_class = self.connection_class
    if hasattr(conn_class, '__init__'):
        classes = [conn_class]
        if hasattr(conn_class, '__bases__'):
            classes += list(conn_class.__bases__)
        for klass in classes:
            if accepts_argument(klass.__init__, 'health_check_interval'):
                break
        else:
            connparams.pop('health_check_interval')
    if conninfo.ssl:
        try:
            connparams.update(conninfo.ssl)
            connparams['connection_class'] = self.connection_class_ssl
        except TypeError:
            pass
    host = connparams['host']
    if '://' in host:
        scheme, _, _, username, password, path, query = _parse_url(host)
        if scheme == 'socket':
            connparams = self._filter_tcp_connparams(**connparams)
            connparams.update({'connection_class': redis.UnixDomainSocketConnection, 'path': '/' + path}, **query)
            connparams.pop('socket_connect_timeout', None)
            connparams.pop('socket_keepalive', None)
            connparams.pop('socket_keepalive_options', None)
        connparams['username'] = username
        connparams['password'] = password
        connparams.pop('host', None)
        connparams.pop('port', None)
    connparams['db'] = self._prepare_virtual_host(connparams.pop('virtual_host', None))
    channel = self
    connection_cls = connparams.get('connection_class') or self.connection_class
    if asynchronous:

        class Connection(connection_cls):

            def disconnect(self, *args):
                super().disconnect(*args)
                channel._on_connection_disconnect(self)
        connection_cls = Connection
    connparams['connection_class'] = connection_cls
    return connparams