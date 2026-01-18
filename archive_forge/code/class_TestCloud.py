from unittest import mock
import uuid
import testtools
from openstack import connection
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
class TestCloud(base.TestCase):

    def test_openstack_cloud(self):
        self.assertIsInstance(self.cloud, connection.Connection)

    def test_endpoint_for(self):
        dns_override = 'https://override.dns.example.com'
        self.cloud.config.config['dns_endpoint_override'] = dns_override
        self.assertEqual('https://compute.example.com/v2.1/', self.cloud.endpoint_for('compute'))
        self.assertEqual('https://internal.compute.example.com/v2.1/', self.cloud.endpoint_for('compute', interface='internal'))
        self.assertIsNone(self.cloud.endpoint_for('compute', region_name='unknown-region'))
        self.assertEqual(dns_override, self.cloud.endpoint_for('dns'))

    def test_connect_as(self):
        project_name = 'test_project'
        self.register_uris([self.get_keystone_v3_token(project_name=project_name), self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': []})])
        c2 = self.cloud.connect_as(project_name=project_name)
        self.assertEqual(c2.list_servers(), [])
        self.assert_calls()

    def test_connect_as_context(self):
        project_name = 'test_project'
        self.register_uris([self.get_keystone_v3_token(project_name=project_name), self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': []})])
        with self.cloud.connect_as(project_name=project_name) as c2:
            self.assertEqual(c2.list_servers(), [])
        self.assert_calls()

    def test_global_request_id(self):
        request_id = uuid.uuid4().hex
        self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': []}, validate=dict(headers={'X-Openstack-Request-Id': request_id}))])
        cloud2 = self.cloud.global_request(request_id)
        self.assertEqual([], cloud2.list_servers())
        self.assert_calls()

    def test_global_request_id_context(self):
        request_id = uuid.uuid4().hex
        self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': []}, validate=dict(headers={'X-Openstack-Request-Id': request_id}))])
        with self.cloud.global_request(request_id) as c2:
            self.assertEqual([], c2.list_servers())
        self.assert_calls()

    def test_iterate_timeout_bad_wait(self):
        with testtools.ExpectedException(exceptions.SDKException, 'Wait value must be an int or float value.'):
            for count in utils.iterate_timeout(1, 'test_iterate_timeout_bad_wait', wait='timeishard'):
                pass

    @mock.patch('time.sleep')
    def test_iterate_timeout_str_wait(self, mock_sleep):
        iter = utils.iterate_timeout(10, 'test_iterate_timeout_str_wait', wait='1.6')
        next(iter)
        next(iter)
        mock_sleep.assert_called_with(1.6)

    @mock.patch('time.sleep')
    def test_iterate_timeout_int_wait(self, mock_sleep):
        iter = utils.iterate_timeout(10, 'test_iterate_timeout_int_wait', wait=1)
        next(iter)
        next(iter)
        mock_sleep.assert_called_with(1.0)

    @mock.patch('time.sleep')
    def test_iterate_timeout_timeout(self, mock_sleep):
        message = 'timeout test'
        with testtools.ExpectedException(exceptions.ResourceTimeout, message):
            for count in utils.iterate_timeout(0.1, message, wait=1):
                pass
        mock_sleep.assert_called_with(1.0)

    def test_range_search(self):
        filters = {'key1': 'min', 'key2': '20'}
        retval = self.cloud.range_search(RANGE_DATA, filters)
        self.assertIsInstance(retval, list)
        self.assertEqual(1, len(retval))
        self.assertEqual([RANGE_DATA[1]], retval)

    def test_range_search_2(self):
        filters = {'key1': '<=2', 'key2': '>10'}
        retval = self.cloud.range_search(RANGE_DATA, filters)
        self.assertIsInstance(retval, list)
        self.assertEqual(2, len(retval))
        self.assertEqual([RANGE_DATA[1], RANGE_DATA[3]], retval)

    def test_range_search_3(self):
        filters = {'key1': '2', 'key2': 'min'}
        retval = self.cloud.range_search(RANGE_DATA, filters)
        self.assertIsInstance(retval, list)
        self.assertEqual(0, len(retval))

    def test_range_search_4(self):
        filters = {'key1': 'max', 'key2': 'min'}
        retval = self.cloud.range_search(RANGE_DATA, filters)
        self.assertIsInstance(retval, list)
        self.assertEqual(0, len(retval))

    def test_range_search_5(self):
        filters = {'key1': 'min', 'key2': 'min'}
        retval = self.cloud.range_search(RANGE_DATA, filters)
        self.assertIsInstance(retval, list)
        self.assertEqual(1, len(retval))
        self.assertEqual([RANGE_DATA[0]], retval)