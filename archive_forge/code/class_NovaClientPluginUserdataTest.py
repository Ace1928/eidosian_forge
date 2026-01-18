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
class NovaClientPluginUserdataTest(NovaClientPluginTestCase):

    def test_build_userdata(self):
        """Tests the build_userdata function."""
        cfg.CONF.set_override('heat_metadata_server_url', 'http://server.test:123')
        cfg.CONF.set_override('instance_connection_is_secure', False)
        cfg.CONF.set_override('instance_connection_https_validate_certificates', False)
        data = self.nova_plugin.build_userdata({})
        self.assertIn('Content-Type: text/cloud-config;', data)
        self.assertIn('Content-Type: text/cloud-boothook;', data)
        self.assertIn('Content-Type: text/part-handler;', data)
        self.assertIn('Content-Type: text/x-cfninitdata;', data)
        self.assertIn('Content-Type: text/x-shellscript;', data)
        self.assertIn('http://server.test:123', data)
        self.assertIn('[Boto]', data)

    def test_build_userdata_without_instance_user(self):
        """Don't add a custom instance user when not requested."""
        cfg.CONF.set_override('heat_metadata_server_url', 'http://server.test:123')
        data = self.nova_plugin.build_userdata({}, instance_user=None)
        self.assertNotIn('user: ', data)
        self.assertNotIn('useradd', data)
        self.assertNotIn('ec2-user', data)

    def test_build_userdata_with_instance_user(self):
        """Add a custom instance user."""
        cfg.CONF.set_override('heat_metadata_server_url', 'http://server.test:123')
        data = self.nova_plugin.build_userdata({}, instance_user='ec2-user')
        self.assertIn('user: ', data)
        self.assertIn('useradd', data)
        self.assertIn('ec2-user', data)

    def test_build_userdata_with_ignition(self):
        metadata = {'os-collect-config': {'heat': {'password': '***'}}}
        userdata = '{"ignition": {"version": "3.0"}, "storage": {"files": []}}'
        ud_format = 'SOFTWARE_CONFIG'
        data = self.nova_plugin.build_userdata(metadata, userdata=userdata, user_data_format=ud_format)
        ig = json.loads(data)
        self.assertEqual('/var/lib/heat-cfntools/cfn-init-data', ig['storage']['files'][0]['path'])
        self.assertEqual('/var/lib/cloud/data/cfn-init-data', ig['storage']['files'][1]['path'])
        self.assertEqual('data:,%7B%22os-collect-config%22%3A%20%7B%22heat%22%3A%20%7B%22password%22%3A%20%22%2A%2A%2A%22%7D%7D%7D', ig['storage']['files'][0]['contents']['source'])

    def test_build_userdata_with_invalid_ignition(self):
        metadata = {'os-collect-config': {'heat': {'password': '***'}}}
        userdata = '{"ignition": {"version": "3.0"}, "storage": []}'
        ud_format = 'SOFTWARE_CONFIG'
        self.assertRaises(ValueError, self.nova_plugin.build_userdata, metadata, userdata=userdata, user_data_format=ud_format)