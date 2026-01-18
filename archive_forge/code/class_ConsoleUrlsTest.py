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
class ConsoleUrlsTest(common.HeatTestCase):
    scenarios = [('novnc', dict(console_type='novnc', res_obj=True)), ('xvpvnc', dict(console_type='xvpvnc', res_obj=True)), ('spice', dict(console_type='spice-html5', res_obj=True)), ('rdp', dict(console_type='rdp-html5', res_obj=True)), ('serial', dict(console_type='serial', res_obj=True)), ('mks', dict(console_type='webmks', res_obj=False))]

    def setUp(self):
        super(ConsoleUrlsTest, self).setUp()
        self.nova_client = mock.Mock()
        con = utils.dummy_context()
        c = con.clients
        self.nova_plugin = c.client_plugin('nova')
        self.patchobject(self.nova_plugin, 'client', return_value=self.nova_client)
        self.server = mock.Mock()
        if self.res_obj:
            self.console_method = getattr(self.server, 'get_console_url')
        else:
            self.console_method = getattr(self.nova_client.servers, 'get_console_url')

    def test_get_console_url(self):
        console = {'console': {'type': self.console_type, 'url': '%s_console_url' % self.console_type}}
        self.console_method.return_value = console
        console_url = self.nova_plugin.get_console_urls(self.server)[self.console_type]
        self.assertEqual(console['console']['url'], console_url)
        self._assert_console_method_called()

    def _assert_console_method_called(self):
        if self.console_type == 'webmks':
            self.console_method.assert_called_once_with(self.server, self.console_type)
        else:
            self.console_method.assert_called_once_with(self.console_type)

    def _test_get_console_url_tolerate_exception(self, msg):
        console_url = self.nova_plugin.get_console_urls(self.server)[self.console_type]
        self._assert_console_method_called()
        self.assertIn(msg, console_url)

    def test_get_console_url_tolerate_unavailable(self):
        msg = 'Unavailable console type %s.' % self.console_type
        self.console_method.side_effect = nova_exceptions.BadRequest(400, message=msg)
        self._test_get_console_url_tolerate_exception(msg)

    def test_get_console_url_tolerate_unsupport(self):
        msg = 'Unsupported console_type "%s"' % self.console_type
        self.console_method.side_effect = nova_exceptions.UnsupportedConsoleType(console_type=self.console_type)
        self._test_get_console_url_tolerate_exception(msg)

    def test_get_console_urls_tolerate_other_400(self):
        exc = nova_exceptions.BadRequest
        self.console_method.side_effect = exc(400, message='spam')
        self._test_get_console_url_tolerate_exception('spam')

    def test_get_console_urls_reraises_other(self):
        exc = Exception
        self.console_method.side_effect = exc('spam')
        self._test_get_console_url_tolerate_exception('spam')