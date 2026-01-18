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
class ClientPluginTest(common.HeatTestCase):

    def test_get_client_option(self):
        con = mock.Mock()
        con.auth_url = 'http://auth.example.com:5000/v2.0'
        con.tenant_id = 'b363706f891f48019483f8bd6503c54b'
        con.auth_token = '3bcc3d3a03f44e3d8377f9247b0ad155'
        c = clients.Clients(con)
        con.clients = c
        plugin = FooClientsPlugin(con)
        cfg.CONF.set_override('ca_file', '/tmp/bar', group='clients_heat')
        cfg.CONF.set_override('ca_file', '/tmp/foo', group='clients')
        cfg.CONF.set_override('endpoint_type', 'internalURL', group='clients')
        self.assertEqual('/tmp/bar', plugin._get_client_option('heat', 'ca_file'))
        self.assertEqual('internalURL', plugin._get_client_option('glance', 'endpoint_type'))
        self.assertEqual('/tmp/foo', plugin._get_client_option('foo', 'ca_file'))

    def test_auth_token(self):
        con = mock.Mock()
        con.auth_token = '1234'
        con.trust_id = None
        c = clients.Clients(con)
        con.clients = c
        plugin = FooClientsPlugin(con)
        self.assertEqual('5678', plugin.auth_token)

    def test_url_for(self):
        con = mock.Mock()
        con.auth_token = '1234'
        con.trust_id = None
        c = clients.Clients(con)
        con.clients = c
        con.keystone_session = mock.Mock(name='keystone_Session')
        con.keystone_session.get_endpoint = mock.Mock(name='get_endpoint')
        con.keystone_session.get_endpoint.return_value = 'http://192.0.2.1/foo'
        plugin = FooClientsPlugin(con)
        self.assertEqual('http://192.0.2.1/foo', plugin.url_for(service_type='foo'))
        self.assertTrue(con.keystone_session.get_endpoint.called)

    @mock.patch.object(generic, 'Token', name='v3_token')
    def test_get_missing_service_catalog(self, mock_v3):

        class FakeKeystone(fake_ks.FakeKeystoneClient):

            def __init__(self):
                super(FakeKeystone, self).__init__()
                self.client = self
                self.version = 'v3'
        self.stub_keystoneclient(fake_client=FakeKeystone())
        con = mock.MagicMock(auth_token='1234', trust_id=None)
        c = clients.Clients(con)
        con.clients = c
        con.keystone_session = mock.Mock(name='keystone_session')
        get_endpoint_side_effects = [keystone_exc.EmptyCatalog(), None, 'http://192.0.2.1/bar']
        con.keystone_session.get_endpoint = mock.Mock(name='get_endpoint', side_effect=get_endpoint_side_effects)
        mock_token_obj = mock.Mock()
        mock_token_obj.get_auth_ref.return_value = {'catalog': 'foo'}
        mock_v3.return_value = mock_token_obj
        plugin = FooClientsPlugin(con)
        self.assertEqual('http://192.0.2.1/bar', plugin.url_for(service_type='bar'))

    @mock.patch.object(generic, 'Token', name='v3_token')
    def test_endpoint_not_found(self, mock_v3):

        class FakeKeystone(fake_ks.FakeKeystoneClient):

            def __init__(self):
                super(FakeKeystone, self).__init__()
                self.client = self
                self.version = 'v3'
        self.stub_keystoneclient(fake_client=FakeKeystone())
        con = mock.MagicMock(auth_token='1234', trust_id=None)
        c = clients.Clients(con)
        con.clients = c
        con.keystone_session = mock.Mock(name='keystone_session')
        get_endpoint_side_effects = [keystone_exc.EmptyCatalog(), None]
        con.keystone_session.get_endpoint = mock.Mock(name='get_endpoint', side_effect=get_endpoint_side_effects)
        mock_token_obj = mock.Mock()
        mock_v3.return_value = mock_token_obj
        mock_access = mock.Mock()
        self.patchobject(mock_token_obj, 'get_access', return_value=mock_access)
        self.patchobject(mock_access, 'has_service_catalog', return_value=False)
        plugin = FooClientsPlugin(con)
        self.assertRaises(keystone_exc.EndpointNotFound, plugin.url_for, service_type='nonexistent')

    def test_abstract_create(self):
        con = mock.Mock()
        c = clients.Clients(con)
        con.clients = c
        self.assertRaises(TypeError, client_plugin.ClientPlugin, c)