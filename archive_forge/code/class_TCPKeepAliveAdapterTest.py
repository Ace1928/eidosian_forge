import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
class TCPKeepAliveAdapterTest(utils.TestCase):

    def setUp(self):
        super(TCPKeepAliveAdapterTest, self).setUp()
        self.init_poolmanager = self.patch(client_session.requests.adapters.HTTPAdapter, 'init_poolmanager')
        self.constructor = self.patch(client_session.TCPKeepAliveAdapter, '__init__', lambda self: None)

    def test_init_poolmanager_with_requests_lesser_than_2_4_1(self):
        self.patch(client_session, 'REQUESTS_VERSION', (2, 4, 0))
        given_adapter = client_session.TCPKeepAliveAdapter()
        given_adapter.init_poolmanager(1, 2, 3)
        self.init_poolmanager.assert_called_once_with(1, 2, 3)

    def test_init_poolmanager_with_basic_options(self):
        self.patch(client_session, 'REQUESTS_VERSION', (2, 4, 1))
        socket = self.patch_socket_with_options(['IPPROTO_TCP', 'TCP_NODELAY', 'SOL_SOCKET', 'SO_KEEPALIVE'])
        given_adapter = client_session.TCPKeepAliveAdapter()
        given_adapter.init_poolmanager(1, 2, 3)
        self.init_poolmanager.assert_called_once_with(1, 2, 3, socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1), (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)])

    def test_init_poolmanager_with_tcp_keepidle(self):
        self.patch(client_session, 'REQUESTS_VERSION', (2, 4, 1))
        socket = self.patch_socket_with_options(['IPPROTO_TCP', 'TCP_NODELAY', 'SOL_SOCKET', 'SO_KEEPALIVE', 'TCP_KEEPIDLE'])
        given_adapter = client_session.TCPKeepAliveAdapter()
        given_adapter.init_poolmanager(1, 2, 3)
        self.init_poolmanager.assert_called_once_with(1, 2, 3, socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1), (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1), (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)])

    def test_init_poolmanager_with_tcp_keepcnt(self):
        self.patch(client_session, 'REQUESTS_VERSION', (2, 4, 1))
        self.patch(client_session.utils, 'is_windows_linux_subsystem', False)
        socket = self.patch_socket_with_options(['IPPROTO_TCP', 'TCP_NODELAY', 'SOL_SOCKET', 'SO_KEEPALIVE', 'TCP_KEEPCNT'])
        given_adapter = client_session.TCPKeepAliveAdapter()
        given_adapter.init_poolmanager(1, 2, 3)
        self.init_poolmanager.assert_called_once_with(1, 2, 3, socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1), (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1), (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 4)])

    def test_init_poolmanager_with_tcp_keepcnt_on_windows(self):
        self.patch(client_session, 'REQUESTS_VERSION', (2, 4, 1))
        self.patch(client_session.utils, 'is_windows_linux_subsystem', True)
        socket = self.patch_socket_with_options(['IPPROTO_TCP', 'TCP_NODELAY', 'SOL_SOCKET', 'SO_KEEPALIVE', 'TCP_KEEPCNT'])
        given_adapter = client_session.TCPKeepAliveAdapter()
        given_adapter.init_poolmanager(1, 2, 3)
        self.init_poolmanager.assert_called_once_with(1, 2, 3, socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1), (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)])

    def test_init_poolmanager_with_tcp_keepintvl(self):
        self.patch(client_session, 'REQUESTS_VERSION', (2, 4, 1))
        socket = self.patch_socket_with_options(['IPPROTO_TCP', 'TCP_NODELAY', 'SOL_SOCKET', 'SO_KEEPALIVE', 'TCP_KEEPINTVL'])
        given_adapter = client_session.TCPKeepAliveAdapter()
        given_adapter.init_poolmanager(1, 2, 3)
        self.init_poolmanager.assert_called_once_with(1, 2, 3, socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1), (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1), (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 15)])

    def test_init_poolmanager_with_given_optionsl(self):
        self.patch(client_session, 'REQUESTS_VERSION', (2, 4, 1))
        given_adapter = client_session.TCPKeepAliveAdapter()
        given_options = object()
        given_adapter.init_poolmanager(1, 2, 3, socket_options=given_options)
        self.init_poolmanager.assert_called_once_with(1, 2, 3, socket_options=given_options)

    def patch_socket_with_options(self, option_names):
        socket = type('socket', (object,), {name: 'socket.' + name for name in option_names})
        return self.patch(client_session, 'socket', socket)

    def patch(self, target, name, *args, **kwargs):
        context = mock.patch.object(target, name, *args, **kwargs)
        patch = context.start()
        self.addCleanup(context.stop)
        return patch