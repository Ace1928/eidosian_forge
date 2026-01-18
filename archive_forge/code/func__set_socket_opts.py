import copy
import os
import socket
import eventlet
import eventlet.wsgi
import greenlet
from paste import deploy
import routes.middleware
import webob.dec
import webob.exc
from oslo_log import log as logging
from oslo_service._i18n import _
from oslo_service import _options
from oslo_service import service
from oslo_service import sslutils
def _set_socket_opts(self, _socket):
    _socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if hasattr(socket, 'TCP_KEEPIDLE'):
        _socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.conf.tcp_keepidle)
    return _socket