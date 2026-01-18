from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import server_remote_console
from openstack.tests.unit import base
class TestServerRemoteConsole(base.TestCase):

    def setUp(self):
        super(TestServerRemoteConsole, self).setUp()
        self.sess = mock.Mock(spec=adapter.Adapter)
        self.sess.default_microversion = '2.9'
        self.resp = mock.Mock()
        self.resp.body = None
        self.resp.json = mock.Mock(return_value=self.resp.body)
        self.resp.status_code = 200
        self.sess = mock.Mock()
        self.sess.post = mock.Mock(return_value=self.resp)
        self.sess._get_connection = mock.Mock(return_value=self.cloud)

    def test_basic(self):
        sot = server_remote_console.ServerRemoteConsole()
        self.assertEqual('remote_console', sot.resource_key)
        self.assertEqual('/servers/%(server_id)s/remote-consoles', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertFalse(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertFalse(sot.allow_list)

    def test_make_it(self):
        sot = server_remote_console.ServerRemoteConsole(**EXAMPLE)
        self.assertEqual(EXAMPLE['url'], sot.url)

    def test_create_type_mks_old(self):
        sot = server_remote_console.ServerRemoteConsole(server_id='fake_server', type='webmks')

        class FakeEndpointData:
            min_microversion = '2'
            max_microversion = '2.5'
        self.sess.get_endpoint_data.return_value = FakeEndpointData()
        self.assertRaises(ValueError, sot.create, self.sess)