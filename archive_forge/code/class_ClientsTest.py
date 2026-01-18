from unittest import mock
from aodhclient import exceptions as aodh_exc
from cinderclient import exceptions as cinder_exc
from glanceclient import exc as glance_exc
from heatclient import client as heatclient
from heatclient import exc as heat_exc
from keystoneauth1 import exceptions as keystone_exc
from keystoneauth1.identity import generic
from manilaclient import exceptions as manila_exc
from mistralclient.api import base as mistral_base
from neutronclient.common import exceptions as neutron_exc
from openstack import exceptions
from oslo_config import cfg
from saharaclient.api import base as sahara_base
from swiftclient import exceptions as swift_exc
from testtools import testcase
from troveclient import client as troveclient
from zaqarclient.transport import errors as zaqar_exc
from heat.common import exception
from heat.engine import clients
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.tests import common
from heat.tests import fakes
from heat.tests.openstack.nova import fakes as fakes_nova
class ClientsTest(common.HeatTestCase):

    def test_bad_cloud_backend(self):
        con = mock.Mock()
        cfg.CONF.set_override('cloud_backend', 'some.weird.object')
        exc = self.assertRaises(exception.Invalid, clients.Clients, con)
        self.assertIn('Invalid cloud_backend setting in heat.conf detected', str(exc))
        cfg.CONF.set_override('cloud_backend', 'heat.engine.clients.Clients')
        exc = self.assertRaises(exception.Invalid, clients.Clients, con)
        self.assertIn('Invalid cloud_backend setting in heat.conf detected', str(exc))

    def test_clients_get_heat_url(self):
        con = mock.Mock()
        con.tenant_id = 'b363706f891f48019483f8bd6503c54b'
        c = clients.Clients(con)
        con.clients = c
        obj = c.client_plugin('heat')
        obj._get_client_option = mock.Mock()
        obj._get_client_option.return_value = None
        obj.url_for = mock.Mock(name='url_for')
        obj.url_for.return_value = 'url_from_keystone'
        self.assertEqual('url_from_keystone', obj.get_heat_url())
        heat_url = 'http://0.0.0.0:8004/v1/%(tenant_id)s'
        obj._get_client_option.return_value = heat_url
        tenant_id = 'b363706f891f48019483f8bd6503c54b'
        result = heat_url % {'tenant_id': tenant_id}
        self.assertEqual(result, obj.get_heat_url())
        obj._get_client_option.return_value = result
        self.assertEqual(result, obj.get_heat_url())

    def _client_cfn_url(self, use_uwsgi=False, use_ipv6=False):
        con = mock.Mock()
        c = clients.Clients(con)
        con.clients = c
        obj = c.client_plugin('heat')
        obj._get_client_option = mock.Mock()
        obj._get_client_option.return_value = None
        obj.url_for = mock.Mock(name='url_for')
        if use_ipv6:
            if use_uwsgi:
                obj.url_for.return_value = 'http://[::1]/heat-api-cfn/v1/'
            else:
                obj.url_for.return_value = 'http://[::1]:8000/v1/'
        elif use_uwsgi:
            obj.url_for.return_value = 'http://0.0.0.0/heat-api-cfn/v1/'
        else:
            obj.url_for.return_value = 'http://0.0.0.0:8000/v1/'
        return obj

    def test_clients_get_heat_cfn_url(self):
        obj = self._client_cfn_url()
        self.assertEqual('http://0.0.0.0:8000/v1/', obj.get_heat_cfn_url())

    def test_clients_get_heat_cfn_metadata_url(self):
        obj = self._client_cfn_url()
        self.assertEqual('http://0.0.0.0:8000/v1/', obj.get_cfn_metadata_server_url())

    def test_clients_get_heat_cfn_metadata_url_conf(self):
        cfg.CONF.set_override('heat_metadata_server_url', 'http://server.test:123')
        obj = self._client_cfn_url()
        self.assertEqual('http://server.test:123/v1/', obj.get_cfn_metadata_server_url())

    @mock.patch.object(heatclient, 'Client')
    def test_clients_heat(self, mock_call):
        self.stub_keystoneclient()
        con = mock.Mock()
        con.auth_url = 'http://auth.example.com:5000/v2.0'
        con.tenant_id = 'b363706f891f48019483f8bd6503c54b'
        con.auth_token = '3bcc3d3a03f44e3d8377f9247b0ad155'
        c = clients.Clients(con)
        con.clients = c
        obj = c.client_plugin('heat')
        obj.url_for = mock.Mock(name='url_for')
        obj.url_for.return_value = 'url_from_keystone'
        obj.client()
        self.assertEqual('url_from_keystone', obj.get_heat_url())

    @mock.patch.object(heatclient, 'Client')
    def test_clients_heat_no_auth_token(self, mock_call):
        con = mock.Mock()
        con.auth_url = 'http://auth.example.com:5000/v2.0'
        con.tenant_id = 'b363706f891f48019483f8bd6503c54b'
        con.auth_token = None
        con.auth_plugin = fakes.FakeAuth(auth_token='anewtoken')
        c = clients.Clients(con)
        con.clients = c
        obj = c.client_plugin('heat')
        obj.url_for = mock.Mock(name='url_for')
        obj.url_for.return_value = 'url_from_keystone'
        self.assertEqual('url_from_keystone', obj.get_heat_url())

    @mock.patch.object(heatclient, 'Client')
    def test_clients_heat_cached(self, mock_call):
        self.stub_auth()
        con = mock.Mock()
        con.auth_url = 'http://auth.example.com:5000/v2.0'
        con.tenant_id = 'b363706f891f48019483f8bd6503c54b'
        con.auth_token = '3bcc3d3a03f44e3d8377f9247b0ad155'
        con.trust_id = None
        c = clients.Clients(con)
        con.clients = c
        obj = c.client_plugin('heat')
        obj.get_heat_url = mock.Mock(name='get_heat_url')
        obj.get_heat_url.return_value = None
        obj.url_for = mock.Mock(name='url_for')
        obj.url_for.return_value = 'url_from_keystone'
        heat = obj.client()
        heat_cached = obj.client()
        self.assertEqual(heat, heat_cached)