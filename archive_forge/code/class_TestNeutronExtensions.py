import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
class TestNeutronExtensions(base.TestCase):

    def test__neutron_extensions(self):
        body = [{'updated': '2014-06-1T10:00:00-00:00', 'name': 'Distributed Virtual Router', 'links': [], 'alias': 'dvr', 'description': 'Enables configuration of Distributed Virtual Routers.'}, {'updated': '2013-07-23T10:00:00-00:00', 'name': 'Allowed Address Pairs', 'links': [], 'alias': 'allowed-address-pairs', 'description': 'Provides allowed address pairs'}]
        self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json=dict(extensions=body))])
        extensions = self.cloud._neutron_extensions()
        self.assertEqual(set(['dvr', 'allowed-address-pairs']), extensions)
        self.assert_calls()

    def test__neutron_extensions_fails(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), status_code=404)])
        with testtools.ExpectedException(exceptions.ResourceNotFound):
            self.cloud._neutron_extensions()
        self.assert_calls()

    def test__has_neutron_extension(self):
        body = [{'updated': '2014-06-1T10:00:00-00:00', 'name': 'Distributed Virtual Router', 'links': [], 'alias': 'dvr', 'description': 'Enables configuration of Distributed Virtual Routers.'}, {'updated': '2013-07-23T10:00:00-00:00', 'name': 'Allowed Address Pairs', 'links': [], 'alias': 'allowed-address-pairs', 'description': 'Provides allowed address pairs'}]
        self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json=dict(extensions=body))])
        self.assertTrue(self.cloud._has_neutron_extension('dvr'))
        self.assert_calls()

    def test__has_neutron_extension_missing(self):
        body = [{'updated': '2014-06-1T10:00:00-00:00', 'name': 'Distributed Virtual Router', 'links': [], 'alias': 'dvr', 'description': 'Enables configuration of Distributed Virtual Routers.'}, {'updated': '2013-07-23T10:00:00-00:00', 'name': 'Allowed Address Pairs', 'links': [], 'alias': 'allowed-address-pairs', 'description': 'Provides allowed address pairs'}]
        self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json=dict(extensions=body))])
        self.assertFalse(self.cloud._has_neutron_extension('invalid'))
        self.assert_calls()