import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack.config import cloud_region
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
class TestFromSession(base.TestCase):
    scenarios = [('no_region', dict(test_region=None)), ('with_region', dict(test_region='RegionOne'))]

    def test_from_session(self):
        config = cloud_region.from_session(self.cloud.session, region_name=self.test_region)
        self.assertEqual(config.name, 'identity.example.com')
        if not self.test_region:
            self.assertIsNone(config.region_name)
        else:
            self.assertEqual(config.region_name, self.test_region)
        server_id = str(uuid.uuid4())
        server_name = self.getUniqueString('name')
        fake_server = fakes.make_fake_server(server_id, server_name)
        self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': [fake_server]})])
        conn = connection.Connection(config=config)
        s = next(conn.compute.servers())
        self.assertEqual(s.id, server_id)
        self.assertEqual(s.name, server_name)
        self.assert_calls()