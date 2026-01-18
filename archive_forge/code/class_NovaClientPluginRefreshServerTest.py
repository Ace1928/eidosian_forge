import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class NovaClientPluginRefreshServerTest(NovaClientPluginTestCase):
    msg = 'ClientException: The server has either erred or is incapable of performing the requested operation.'
    scenarios = [('successful_refresh', dict(value=None, e_raise=False)), ('overlimit_error', dict(value=nova_exceptions.OverLimit(413, 'limit reached'), e_raise=False)), ('500_error', dict(value=nova_exceptions.ClientException(500, msg), e_raise=False)), ('503_error', dict(value=nova_exceptions.ClientException(503, msg), e_raise=False)), ('unhandled_exception', dict(value=nova_exceptions.ClientException(501, msg), e_raise=True))]

    def test_refresh(self):
        server = mock.MagicMock()
        server.get.side_effect = [self.value]
        if self.e_raise:
            self.assertRaises(nova_exceptions.ClientException, self.nova_plugin.refresh_server, server)
        else:
            self.assertIsNone(self.nova_plugin.refresh_server(server))
        server.get.assert_called_once_with()