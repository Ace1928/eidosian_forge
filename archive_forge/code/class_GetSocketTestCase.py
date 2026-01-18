import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
class GetSocketTestCase(test_utils.BaseTestCase):

    def setUp(self):
        super(GetSocketTestCase, self).setUp()
        self.useFixture(fixtures.MonkeyPatch('glance.common.wsgi.get_bind_addr', lambda x: ('192.168.0.13', 1234)))
        addr_info_list = [(2, 1, 6, '', ('192.168.0.13', 80)), (2, 2, 17, '', ('192.168.0.13', 80)), (2, 3, 0, '', ('192.168.0.13', 80))]
        self.useFixture(fixtures.MonkeyPatch('glance.common.wsgi.socket.getaddrinfo', lambda *x: addr_info_list))
        self.useFixture(fixtures.MonkeyPatch('glance.common.wsgi.time.time', mock.Mock(side_effect=[0, 1, 5, 10, 20, 35])))
        self.useFixture(fixtures.MonkeyPatch('glance.common.wsgi.utils.validate_key_cert', lambda *x: None))
        wsgi.CONF.tcp_keepidle = 600

    @mock.patch.object(prefetcher, 'Prefetcher')
    def test_correct_configure_socket(self, mock_prefetcher):
        mock_socket = mock.Mock()
        self.useFixture(fixtures.MonkeyPatch('glance.common.wsgi.eventlet.listen', lambda *x, **y: mock_socket))
        server = wsgi.Server()
        server.default_port = 1234
        server.configure_socket()
        self.assertIn(mock.call.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1), mock_socket.mock_calls)
        self.assertIn(mock.call.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1), mock_socket.mock_calls)
        if hasattr(socket, 'TCP_KEEPIDLE'):
            self.assertIn(mock.call.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, wsgi.CONF.tcp_keepidle), mock_socket.mock_calls)

    def test_get_socket_with_bind_problems(self):
        self.useFixture(fixtures.MonkeyPatch('glance.common.wsgi.eventlet.listen', mock.Mock(side_effect=[wsgi.socket.error(socket.errno.EADDRINUSE)] * 3 + [None])))
        self.assertRaises(RuntimeError, wsgi.get_socket, 1234)

    def test_get_socket_with_unexpected_socket_errno(self):
        self.useFixture(fixtures.MonkeyPatch('glance.common.wsgi.eventlet.listen', mock.Mock(side_effect=wsgi.socket.error(socket.errno.ENOMEM))))
        self.assertRaises(wsgi.socket.error, wsgi.get_socket, 1234)