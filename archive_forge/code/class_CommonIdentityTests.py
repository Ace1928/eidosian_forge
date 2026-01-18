import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
class CommonIdentityTests(metaclass=abc.ABCMeta):
    PROJECT_ID = uuid.uuid4().hex
    TEST_ROOT_URL = 'http://127.0.0.1:5000/'
    TEST_ROOT_ADMIN_URL = 'http://127.0.0.1:35357/'
    TEST_COMPUTE_BASE = 'https://compute.example.com'
    TEST_COMPUTE_PUBLIC = TEST_COMPUTE_BASE + '/nova/public'
    TEST_COMPUTE_INTERNAL = TEST_COMPUTE_BASE + '/nova/internal'
    TEST_COMPUTE_ADMIN = TEST_COMPUTE_BASE + '/nova/admin'
    TEST_VOLUME = FakeServiceEndpoints(base_url='https://block-storage.example.com', versions=['v3', 'v2'], project_id=PROJECT_ID)
    TEST_BAREMETAL_BASE = 'https://baremetal.example.com'
    TEST_BAREMETAL_INTERNAL = TEST_BAREMETAL_BASE + '/internal'
    TEST_PASS = uuid.uuid4().hex

    def setUp(self):
        super(CommonIdentityTests, self).setUp()
        self.TEST_URL = '%s%s' % (self.TEST_ROOT_URL, self.version)
        self.TEST_ADMIN_URL = '%s%s' % (self.TEST_ROOT_ADMIN_URL, self.version)
        self.TEST_DISCOVERY = fixture.DiscoveryList(href=self.TEST_ROOT_URL)
        self.stub_auth_data()

    @abc.abstractmethod
    def create_auth_plugin(self, **kwargs):
        """Create an auth plugin that makes sense for the auth data.

        It doesn't really matter what auth mechanism is used but it should be
        appropriate to the API version.
        """

    @abc.abstractmethod
    def get_auth_data(self, **kwargs):
        """Return fake authentication data.

        This should register a valid token response and ensure that the compute
        endpoints are set to TEST_COMPUTE_PUBLIC, _INTERNAL and _ADMIN.
        """

    def stub_auth_data(self, **kwargs):
        token = self.get_auth_data(**kwargs)
        self.user_id = token.user_id
        try:
            self.project_id = token.project_id
        except AttributeError:
            self.project_id = token.tenant_id
        self.stub_auth(json=token)

    @property
    @abc.abstractmethod
    def version(self):
        """The API version being tested."""

    def test_discovering(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_nova_microversion(href=self.TEST_COMPUTE_ADMIN, id='v2.1', status='CURRENT', min_version='2.1', version='2.38')
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
        body = 'SUCCESS'
        self.stub_url('GET', ['path'], text=body, base_url=self.TEST_COMPUTE_ADMIN)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        resp = s.get('/path', endpoint_filter={'service_type': 'compute', 'interface': 'admin', 'version': '2.1'})
        self.assertEqual(200, resp.status_code)
        self.assertEqual(body, resp.text)
        new_body = 'SC SUCCESS'
        self.stub_url('GET', ['path'], base_url=self.TEST_COMPUTE_ADMIN, text=new_body)
        resp = s.get('/path', endpoint_filter={'service_type': 'compute', 'interface': 'admin'})
        self.assertEqual(200, resp.status_code)
        self.assertEqual(new_body, resp.text)

    def test_discovery_uses_provided_session_cache(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_nova_microversion(href=self.TEST_COMPUTE_ADMIN, id='v2.1', status='CURRENT', min_version='2.1', version='2.38')
        resps = [{'json': disc}, {'status_code': 500}]
        self.requests_mock.get(self.TEST_COMPUTE_ADMIN, resps)
        body = 'SUCCESS'
        self.stub_url('GET', ['path'], text=body, base_url=self.TEST_COMPUTE_ADMIN)
        cache = {}
        s = session.Session(discovery_cache=cache)
        a = self.create_auth_plugin()
        b = self.create_auth_plugin()
        for auth in (a, b):
            resp = s.get('/path', auth=auth, endpoint_filter={'service_type': 'compute', 'interface': 'admin', 'version': '2.1'})
            self.assertEqual(200, resp.status_code)
            self.assertEqual(body, resp.text)
        self.assertIn(self.TEST_COMPUTE_ADMIN, cache.keys())

    def test_discovery_uses_session_cache(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_nova_microversion(href=self.TEST_COMPUTE_ADMIN, id='v2.1', status='CURRENT', min_version='2.1', version='2.38')
        resps = [{'json': disc}, {'status_code': 500}]
        self.requests_mock.get(self.TEST_COMPUTE_ADMIN, resps)
        body = 'SUCCESS'
        self.stub_url('GET', ['path'], base_url=self.TEST_COMPUTE_ADMIN, text=body)
        filter = {'service_type': 'compute', 'interface': 'admin', 'version': '2.1'}
        sess = session.Session()
        sess.get('/path', auth=self.create_auth_plugin(), endpoint_filter=filter)
        self.assertIn(self.TEST_COMPUTE_ADMIN, sess._discovery_cache.keys())
        a = self.create_auth_plugin()
        b = self.create_auth_plugin()
        for auth in (a, b):
            resp = sess.get('/path', auth=auth, endpoint_filter=filter)
            self.assertEqual(200, resp.status_code)
            self.assertEqual(body, resp.text)

    def test_discovery_uses_plugin_cache(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_nova_microversion(href=self.TEST_COMPUTE_ADMIN, id='v2.1', status='CURRENT', min_version='2.1', version='2.38')
        resps = [{'json': disc}, {'status_code': 500}]
        self.requests_mock.get(self.TEST_COMPUTE_ADMIN, resps)
        body = 'SUCCESS'
        self.stub_url('GET', ['path'], base_url=self.TEST_COMPUTE_ADMIN, text=body)
        sa = session.Session()
        sb = session.Session()
        auth = self.create_auth_plugin()
        for sess in (sa, sb):
            resp = sess.get('/path', auth=auth, endpoint_filter={'service_type': 'compute', 'interface': 'admin', 'version': '2.1'})
            self.assertEqual(200, resp.status_code)
            self.assertEqual(body, resp.text)

    def test_discovery_uses_session_plugin_cache(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_nova_microversion(href=self.TEST_COMPUTE_ADMIN, id='v2.1', status='CURRENT', min_version='2.1', version='2.38')
        resps = [{'json': disc}, {'status_code': 500}]
        self.requests_mock.get(self.TEST_COMPUTE_ADMIN, resps)
        body = 'SUCCESS'
        self.stub_url('GET', ['path'], base_url=self.TEST_COMPUTE_ADMIN, text=body)
        filter = {'service_type': 'compute', 'interface': 'admin', 'version': '2.1'}
        plugin = self.create_auth_plugin()
        session.Session().get('/path', auth=plugin, endpoint_filter=filter)
        self.assertIn(self.TEST_COMPUTE_ADMIN, plugin._discovery_cache.keys())
        sess = session.Session(auth=plugin)
        for auth in (plugin, self.create_auth_plugin()):
            resp = sess.get('/path', auth=auth, endpoint_filter=filter)
            self.assertEqual(200, resp.status_code)
            self.assertEqual(body, resp.text)

    def test_direct_discovery_provided_plugin_cache(self):
        resps = [{'json': self.TEST_DISCOVERY}, {'status_code': 500}]
        self.requests_mock.get(self.TEST_COMPUTE_ADMIN, resps)
        sa = session.Session()
        sb = session.Session()
        discovery_cache = {}
        expected_url = self.TEST_COMPUTE_ADMIN + '/v2.0'
        for sess in (sa, sb):
            disc = discover.get_discovery(sess, self.TEST_COMPUTE_ADMIN, cache=discovery_cache)
            url = disc.url_for(('2', '0'))
            self.assertEqual(expected_url, url)
        self.assertIn(self.TEST_COMPUTE_ADMIN, discovery_cache.keys())

    def test_discovery_trailing_slash(self):
        self.requests_mock.get('https://example.com', [{'json': self.TEST_DISCOVERY}, {'status_code': 500}])
        sess = session.Session()
        discovery_cache = {}
        expected_url = 'https://example.com/v2.0'
        for test_endpoint in ('https://example.com', 'https://example.com/'):
            disc = discover.get_discovery(sess, test_endpoint, cache=discovery_cache)
            url = disc.url_for(('2', '0'))
            self.assertEqual(expected_url, url)
        self.assertIn('https://example.com', discovery_cache.keys())
        self.assertNotIn('https://example.com/', discovery_cache.keys())

    def test_discovering_with_no_data(self):
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, status_code=400)
        body = 'SUCCESS'
        self.stub_url('GET', ['path'], base_url=self.TEST_COMPUTE_ADMIN, text=body, status_code=200)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        resp = s.get('/path', endpoint_filter={'service_type': 'compute', 'interface': 'admin', 'version': self.version})
        self.assertEqual(200, resp.status_code)
        self.assertEqual(body, resp.text)

    def test_direct_discovering_with_no_data(self):
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, status_code=400)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        self.assertRaises(exceptions.BadRequest, discover.get_discovery, s, self.TEST_COMPUTE_ADMIN)

    def test_discovering_with_relative_link(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_v2('v2.0')
        disc.add_v3('v3')
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        endpoint_v2 = s.get_endpoint(service_type='compute', interface='admin', version=(2, 0))
        endpoint_v3 = s.get_endpoint(service_type='compute', interface='admin', version=(3, 0))
        self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v2.0', endpoint_v2)
        self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v3', endpoint_v3)

    def test_direct_discovering(self):
        v2_compute = self.TEST_COMPUTE_ADMIN + '/v2.0'
        v3_compute = self.TEST_COMPUTE_ADMIN + '/v3'
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_v2(v2_compute)
        disc.add_v3(v3_compute)
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        catalog_url = s.get_endpoint(service_type='compute', interface='admin')
        disc = discover.get_discovery(s, catalog_url)
        url_v2 = disc.url_for(('2', '0'))
        url_v3 = disc.url_for(('3', '0'))
        self.assertEqual(v2_compute, url_v2)
        self.assertEqual(v3_compute, url_v3)
        url_v2 = disc.url_for('2.0')
        url_v3 = disc.url_for('3.0')
        self.assertEqual(v2_compute, url_v2)
        self.assertEqual(v3_compute, url_v3)

    def test_discovering_version_no_discovery(self):
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        version = s.get_api_major_version(service_type='volumev2', interface='admin')
        self.assertEqual((2, 0), version)

    def test_discovering_version_with_discovery(self):
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        v2_compute = self.TEST_COMPUTE_ADMIN + '/v2.0'
        v3_compute = self.TEST_COMPUTE_ADMIN + '/v3'
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_v2(v2_compute)
        disc.add_v3(v3_compute)
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
        version = s.get_api_major_version(service_type='compute', interface='admin')
        self.assertEqual((3, 0), version)
        self.assertEqual(self.requests_mock.request_history[-1].url, self.TEST_COMPUTE_ADMIN)

    def test_direct_discovering_with_relative_link(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_v2('v2.0')
        disc.add_v3('v3')
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        catalog_url = s.get_endpoint(service_type='compute', interface='admin')
        disc = discover.get_discovery(s, catalog_url)
        url_v2 = disc.url_for(('2', '0'))
        url_v3 = disc.url_for(('3', '0'))
        self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v2.0', url_v2)
        self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v3', url_v3)
        url_v2 = disc.url_for('2.0')
        url_v3 = disc.url_for('3.0')
        self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v2.0', url_v2)
        self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v3', url_v3)

    def test_discovering_with_relative_anchored_link(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_v2('/v2.0')
        disc.add_v3('/v3')
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        endpoint_v2 = s.get_endpoint(service_type='compute', interface='admin', version=(2, 0))
        endpoint_v3 = s.get_endpoint(service_type='compute', interface='admin', version=(3, 0))
        self.assertEqual(self.TEST_COMPUTE_BASE + '/v2.0', endpoint_v2)
        self.assertEqual(self.TEST_COMPUTE_BASE + '/v3', endpoint_v3)

    def test_discovering_with_protocol_relative(self):
        path = self.TEST_COMPUTE_ADMIN[self.TEST_COMPUTE_ADMIN.find(':') + 1:]
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_v2(path + '/v2.0')
        disc.add_v3(path + '/v3')
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        endpoint_v2 = s.get_endpoint(service_type='compute', interface='admin', version=(2, 0))
        endpoint_v3 = s.get_endpoint(service_type='compute', interface='admin', version=(3, 0))
        self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v2.0', endpoint_v2)
        self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v3', endpoint_v3)

    def test_discovering_when_version_missing(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_v2('v2.0')
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        endpoint_v2 = s.get_endpoint(service_type='compute', interface='admin', version=(2, 0))
        endpoint_v3 = s.get_endpoint(service_type='compute', interface='admin', version=(3, 0))
        self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v2.0', endpoint_v2)
        self.assertIsNone(endpoint_v3)

    def test_endpoint_data_no_version(self):
        path = self.TEST_COMPUTE_ADMIN[self.TEST_COMPUTE_ADMIN.find(':') + 1:]
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_v2(path + '/v2.0')
        disc.add_v3(path + '/v3')
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        data = a.get_endpoint_data(session=s, service_type='compute', interface='admin')
        self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v3', data.url)
        self.assertEqual((3, 0), data.api_version)

    def test_get_all_version_data_all_interfaces(self):
        for interface in ('public', 'internal', 'admin'):
            disc = fixture.DiscoveryList(v2=False, v3=False)
            disc.add_nova_microversion(href=getattr(self.TEST_VOLUME.versions['v3'].discovery, interface), id='v3.0', status='CURRENT', min_version='3.0', version='3.20')
            disc.add_nova_microversion(href=getattr(self.TEST_VOLUME.versions['v2'].discovery, interface), id='v2.0', status='SUPPORTED')
            self.stub_url('GET', [], base_url=getattr(self.TEST_VOLUME.unversioned, interface) + '/', json=disc)
        for url in (self.TEST_COMPUTE_PUBLIC, self.TEST_COMPUTE_INTERNAL, self.TEST_COMPUTE_ADMIN):
            disc = fixture.DiscoveryList(v2=False, v3=False)
            disc.add_microversion(href=url, id='v2')
            disc.add_microversion(href=url, id='v2.1', min_version='2.1', max_version='2.35')
            self.stub_url('GET', [], base_url=url, json=disc)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        identity_endpoint = 'http://127.0.0.1:35357/{}/'.format(self.version)
        data = s.get_all_version_data(interface=None)
        self.assertEqual({'RegionOne': {'admin': {'block-storage': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'SUPPORTED', 'status': 'SUPPORTED', 'url': 'https://block-storage.example.com/admin/v2', 'version': '2.0'}, {'collection': None, 'max_microversion': '3.20', 'min_microversion': '3.0', 'next_min_version': None, 'not_before': None, 'raw_status': 'CURRENT', 'status': 'CURRENT', 'url': 'https://block-storage.example.com/admin/v3', 'version': '3.0'}], 'compute': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/admin', 'version': '2.0'}, {'collection': None, 'max_microversion': '2.35', 'min_microversion': '2.1', 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/admin', 'version': '2.1'}], 'identity': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': None, 'status': 'CURRENT', 'url': identity_endpoint, 'version': self.discovery_version}]}, 'internal': {'baremetal': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': None, 'status': 'CURRENT', 'url': 'https://baremetal.example.com/internal/', 'version': None}], 'block-storage': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'SUPPORTED', 'status': 'SUPPORTED', 'url': 'https://block-storage.example.com/internal/v2', 'version': '2.0'}, {'collection': None, 'max_microversion': '3.20', 'min_microversion': '3.0', 'next_min_version': None, 'not_before': None, 'raw_status': 'CURRENT', 'status': 'CURRENT', 'url': 'https://block-storage.example.com/internal/v3', 'version': '3.0'}], 'compute': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/internal', 'version': '2.0'}, {'collection': None, 'max_microversion': '2.35', 'min_microversion': '2.1', 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/internal', 'version': '2.1'}]}, 'public': {'block-storage': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'SUPPORTED', 'status': 'SUPPORTED', 'url': 'https://block-storage.example.com/public/v2', 'version': '2.0'}, {'collection': None, 'max_microversion': '3.20', 'min_microversion': '3.0', 'next_min_version': None, 'not_before': None, 'raw_status': 'CURRENT', 'status': 'CURRENT', 'url': 'https://block-storage.example.com/public/v3', 'version': '3.0'}], 'compute': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.0'}, {'collection': None, 'max_microversion': '2.35', 'min_microversion': '2.1', 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.1'}]}}}, data)

    def test_get_all_version_data(self):
        cinder_disc = fixture.DiscoveryList(v2=False, v3=False)
        cinder_disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v3'].discovery.public, id='v3.0', status='CURRENT', min_version='3.0', version='3.20')
        cinder_disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v2'].discovery.public, id='v2.0', status='SUPPORTED')
        self.stub_url('GET', [], base_url=self.TEST_VOLUME.unversioned.public + '/', json=cinder_disc)
        nova_disc = fixture.DiscoveryList(v2=False, v3=False)
        nova_disc.add_microversion(href=self.TEST_COMPUTE_PUBLIC, id='v2')
        nova_disc.add_microversion(href=self.TEST_COMPUTE_PUBLIC, id='v2.1', min_version='2.1', max_version='2.35')
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_PUBLIC, json=nova_disc)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        data = s.get_all_version_data(interface='public')
        self.assertEqual({'RegionOne': {'public': {'block-storage': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'SUPPORTED', 'status': 'SUPPORTED', 'url': 'https://block-storage.example.com/public/v2', 'version': '2.0'}, {'collection': None, 'max_microversion': '3.20', 'min_microversion': '3.0', 'next_min_version': None, 'not_before': None, 'raw_status': 'CURRENT', 'status': 'CURRENT', 'url': 'https://block-storage.example.com/public/v3', 'version': '3.0'}], 'compute': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.0'}, {'collection': None, 'max_microversion': '2.35', 'min_microversion': '2.1', 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.1'}]}}}, data)

    def test_get_all_version_data_by_service_type(self):
        nova_disc = fixture.DiscoveryList(v2=False, v3=False)
        nova_disc.add_microversion(href=self.TEST_COMPUTE_PUBLIC, id='v2')
        nova_disc.add_microversion(href=self.TEST_COMPUTE_PUBLIC, id='v2.1', min_version='2.1', max_version='2.35')
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_PUBLIC, json=nova_disc)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        data = s.get_all_version_data(interface='public', service_type='compute')
        self.assertEqual({'RegionOne': {'public': {'compute': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.0'}, {'collection': None, 'max_microversion': '2.35', 'min_microversion': '2.1', 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.1'}]}}}, data)

    def test_get_all_version_data_adapter(self):
        nova_disc = fixture.DiscoveryList(v2=False, v3=False)
        nova_disc.add_microversion(href=self.TEST_COMPUTE_PUBLIC, id='v2')
        nova_disc.add_microversion(href=self.TEST_COMPUTE_PUBLIC, id='v2.1', min_version='2.1', max_version='2.35')
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_PUBLIC, json=nova_disc)
        s = session.Session(auth=self.create_auth_plugin())
        a = adapter.Adapter(session=s, service_type='compute')
        data = a.get_all_version_data()
        self.assertEqual({'RegionOne': {'public': {'compute': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.0'}, {'collection': None, 'max_microversion': '2.35', 'min_microversion': '2.1', 'next_min_version': None, 'not_before': None, 'raw_status': 'stable', 'status': 'CURRENT', 'url': 'https://compute.example.com/nova/public', 'version': '2.1'}]}}}, data)

    def test_get_all_version_data_service_alias(self):
        cinder_disc = fixture.DiscoveryList(v2=False, v3=False)
        cinder_disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v3'].discovery.public, id='v3.0', status='CURRENT', min_version='3.0', version='3.20')
        cinder_disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v2'].discovery.public, id='v2.0', status='SUPPORTED')
        self.stub_url('GET', [], base_url=self.TEST_VOLUME.unversioned.public + '/', json=cinder_disc)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        data = s.get_all_version_data(interface='public', service_type='block-store')
        self.assertEqual({'RegionOne': {'public': {'block-storage': [{'collection': None, 'max_microversion': None, 'min_microversion': None, 'next_min_version': None, 'not_before': None, 'raw_status': 'SUPPORTED', 'status': 'SUPPORTED', 'url': 'https://block-storage.example.com/public/v2', 'version': '2.0'}, {'collection': None, 'max_microversion': '3.20', 'min_microversion': '3.0', 'next_min_version': None, 'not_before': None, 'raw_status': 'CURRENT', 'status': 'CURRENT', 'url': 'https://block-storage.example.com/public/v3', 'version': '3.0'}]}}}, data)

    def test_endpoint_data_no_version_no_discovery(self):
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        data = a.get_endpoint_data(session=s, service_type='compute', interface='admin', discover_versions=False)
        self.assertEqual(self.TEST_COMPUTE_ADMIN, data.url)
        self.assertIsNone(data.api_version)

    def test_endpoint_data_version_url_no_discovery(self):
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        data = a.get_endpoint_data(session=s, service_type='volumev3', interface='admin', discover_versions=False)
        self.assertEqual(self.TEST_VOLUME.versions['v3'].service.admin, data.url)
        self.assertEqual((3, 0), data.api_version)

    def test_endpoint_no_version(self):
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        data = a.get_endpoint(session=s, service_type='compute', interface='admin')
        self.assertEqual(self.TEST_COMPUTE_ADMIN, data)

    def test_endpoint_data_relative_version(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_v2('v2.0')
        disc.add_v3('v3')
        self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        data_v2 = a.get_endpoint_data(session=s, service_type='compute', interface='admin', min_version=(2, 0), max_version=(2, discover.LATEST))
        data_v3 = a.get_endpoint_data(session=s, service_type='compute', interface='admin', min_version=(3, 0), max_version=(3, discover.LATEST))
        self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v2.0', data_v2.url)
        self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v3', data_v3.url)

    def test_get_versioned_data(self):
        v2_compute = self.TEST_COMPUTE_ADMIN + '/v2.0'
        v3_compute = self.TEST_COMPUTE_ADMIN + '/v3'
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_v2(v2_compute)
        disc.add_v3(v3_compute)
        resps = [{'json': disc}, {'status_code': 500}]
        self.requests_mock.get(self.TEST_COMPUTE_ADMIN, resps)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        data = a.get_endpoint_data(session=s, service_type='compute', interface='admin')
        self.assertEqual(v3_compute, data.url)
        v2_data = data.get_versioned_data(s, min_version='2.0', max_version='2.latest')
        self.assertEqual(v2_compute, v2_data.url)
        self.assertEqual(v2_compute, v2_data.service_url)
        self.assertEqual(self.TEST_COMPUTE_ADMIN, v2_data.catalog_url)
        for vkwargs in (dict(min_version='3.0', max_version='3.latest'), dict(min_version='2.0', max_version='3.latest'), dict(min_version='2.0', max_version='latest'), dict(min_version='2.0'), dict()):
            v3_data = data.get_versioned_data(s, **vkwargs)
            self.assertEqual(v3_compute, v3_data.url)
            self.assertEqual(v3_compute, v3_data.service_url)
            self.assertEqual(self.TEST_COMPUTE_ADMIN, v3_data.catalog_url)

    def test_get_current_versioned_data(self):
        v2_compute = self.TEST_COMPUTE_ADMIN + '/v2.0'
        v3_compute = self.TEST_COMPUTE_ADMIN + '/v3'
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_v2(v2_compute)
        disc.add_v3(v3_compute)
        resps = [{'json': disc}, {'status_code': 500}]
        self.requests_mock.get(self.TEST_COMPUTE_ADMIN, resps)
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        data = a.get_endpoint_data(session=s, service_type='compute', interface='admin')
        self.assertEqual(v3_compute, data.url)
        v3_data = data.get_current_versioned_data(s)
        self.assertEqual(v3_compute, v3_data.url)
        self.assertEqual(v3_compute, v3_data.service_url)
        self.assertEqual(self.TEST_COMPUTE_ADMIN, v3_data.catalog_url)
        self.assertEqual((3, 0), v3_data.api_version)
        self.assertIsNone(v3_data.min_microversion)
        self.assertIsNone(v3_data.max_microversion)

    def test_interface_list(self):
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        ep = s.get_endpoint(service_type='baremetal', interface=['internal', 'public'])
        self.assertEqual(ep, self.TEST_BAREMETAL_INTERNAL)
        ep = s.get_endpoint(service_type='baremetal', interface=['public', 'internal'])
        self.assertEqual(ep, self.TEST_BAREMETAL_INTERNAL)
        ep = s.get_endpoint(service_type='compute', interface=['internal', 'public'])
        self.assertEqual(ep, self.TEST_COMPUTE_INTERNAL)
        ep = s.get_endpoint(service_type='compute', interface=['public', 'internal'])
        self.assertEqual(ep, self.TEST_COMPUTE_PUBLIC)

    def test_get_versioned_data_volume_project_id(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v3'].discovery.public, id='v3.0', status='CURRENT', min_version='3.0', version='3.20')
        disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v2'].discovery.public, id='v2.0', status='SUPPORTED')
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        endpoint = a.get_endpoint(session=s, service_type='volumev3', interface='public', version='3.0')
        self.assertEqual(self.TEST_VOLUME.catalog.public, endpoint)
        resps = [{'json': disc}, {'status_code': 500}]
        self.requests_mock.get(self.TEST_VOLUME.versions['v3'].discovery.public + '/', resps)
        data = a.get_endpoint_data(session=s, service_type='volumev3', interface='public')
        self.assertEqual(self.TEST_VOLUME.versions['v3'].service.public, data.url)
        v3_data = data.get_versioned_data(s, min_version='3.0', max_version='3.latest', project_id=self.project_id)
        self.assertEqual(self.TEST_VOLUME.versions['v3'].service.public, v3_data.url)
        self.assertEqual(self.TEST_VOLUME.catalog.public, v3_data.catalog_url)
        self.assertEqual((3, 0), v3_data.min_microversion)
        self.assertEqual((3, 20), v3_data.max_microversion)
        self.assertEqual(self.TEST_VOLUME.versions['v3'].service.public, v3_data.service_url)
        self.requests_mock.get(self.TEST_VOLUME.unversioned.public, resps)
        v2_data = data.get_versioned_data(s, min_version='2.0', max_version='2.latest', project_id=self.project_id)
        self.assertEqual(self.TEST_VOLUME.versions['v2'].service.public, v2_data.url)
        self.assertEqual(self.TEST_VOLUME.versions['v2'].service.public, v2_data.service_url)
        self.assertEqual(self.TEST_VOLUME.catalog.public, v2_data.catalog_url)
        self.assertIsNone(v2_data.min_microversion)
        self.assertIsNone(v2_data.max_microversion)

    def test_get_versioned_data_volume_project_id_unversioned_first(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v3'].discovery.public, id='v3.0', status='CURRENT', min_version='3.0', version='3.20')
        disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v2'].discovery.public, id='v2.0', status='SUPPORTED')
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        endpoint = a.get_endpoint(session=s, service_type='volumev3', interface='public', version='3.0')
        self.assertEqual(self.TEST_VOLUME.catalog.public, endpoint)
        resps = [{'json': disc}, {'status_code': 500}]
        self.requests_mock.get(self.TEST_VOLUME.unversioned.public + '/', resps)
        v2_data = s.get_endpoint_data(service_type='block-storage', interface='public', min_version='2.0', max_version='2.latest', project_id=self.project_id)
        self.assertEqual(self.TEST_VOLUME.versions['v2'].service.public, v2_data.url)
        self.assertEqual(self.TEST_VOLUME.versions['v2'].service.public, v2_data.service_url)
        self.assertEqual(self.TEST_VOLUME.catalog.public, v2_data.catalog_url)
        self.assertIsNone(v2_data.min_microversion)
        self.assertIsNone(v2_data.max_microversion)
        v3_data = v2_data.get_versioned_data(s, min_version='3.0', max_version='3.latest', project_id=self.project_id)
        self.assertEqual(self.TEST_VOLUME.versions['v3'].service.public, v3_data.url)
        self.assertEqual(self.TEST_VOLUME.catalog.public, v3_data.catalog_url)
        self.assertEqual((3, 0), v3_data.min_microversion)
        self.assertEqual((3, 20), v3_data.max_microversion)
        self.assertEqual(self.TEST_VOLUME.versions['v3'].service.public, v3_data.service_url)

    def test_trailing_slash_on_computed_endpoint(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v3'].discovery.public, id='v3.0', status='CURRENT', min_version='3.0', version='3.20')
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        self.requests_mock.get(self.TEST_VOLUME.unversioned.public + '/', json=disc)
        s.get_endpoint_data(service_type='block-storage', interface='public', min_version='2.0', max_version='2.latest', project_id=self.project_id)
        self.assertTrue(self.requests_mock.request_history[-1].url.endswith('/'))

    def test_no_trailing_slash_on_catalog_endpoint(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_nova_microversion(href=self.TEST_COMPUTE_PUBLIC, id='v2.1', status='CURRENT', min_version='2.1', version='2.38')
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        self.requests_mock.get(self.TEST_COMPUTE_PUBLIC, json=disc)
        s.get_endpoint_data(service_type='compute', interface='public', min_version='2.1', max_version='2.latest')
        self.assertFalse(self.requests_mock.request_history[-1].url.endswith('/'))

    def test_broken_discovery_endpoint(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        disc.add_nova_microversion(href='http://internal.example.com', id='v2.1', status='CURRENT', min_version='2.1', version='2.38')
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        self.requests_mock.get(self.TEST_COMPUTE_PUBLIC, json=disc)
        data = s.get_endpoint_data(service_type='compute', interface='public', min_version='2.1', max_version='2.latest')
        self.assertTrue(data.url, self.TEST_COMPUTE_PUBLIC + '/v2.1')

    def test_asking_for_auth_endpoint_ignores_checks(self):
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        auth_url = s.get_endpoint(service_type='compute', interface=plugin.AUTH_INTERFACE)
        self.assertEqual(self.TEST_URL, auth_url)

    def _create_expired_auth_plugin(self, **kwargs):
        expires = _utils.before_utcnow(minutes=20)
        expired_token = self.get_auth_data(expires=expires)
        expired_auth_ref = access.create(body=expired_token)
        a = self.create_auth_plugin(**kwargs)
        a.auth_ref = expired_auth_ref
        return a

    def test_reauthenticate(self):
        a = self._create_expired_auth_plugin()
        expired_auth_ref = a.auth_ref
        s = session.Session(auth=a)
        self.assertIsNot(expired_auth_ref, a.get_access(s))

    def test_no_reauthenticate(self):
        a = self._create_expired_auth_plugin(reauthenticate=False)
        expired_auth_ref = a.auth_ref
        s = session.Session(auth=a)
        self.assertIs(expired_auth_ref, a.get_access(s))

    def test_invalidate(self):
        a = self.create_auth_plugin()
        s = session.Session(auth=a)
        s.get_auth_headers()
        self.assertTrue(a.auth_ref)
        self.assertTrue(a.invalidate())
        self.assertIsNone(a.auth_ref)
        self.assertFalse(a.invalidate())

    def test_get_auth_properties(self):
        a = self.create_auth_plugin()
        s = session.Session()
        self.assertEqual(self.user_id, a.get_user_id(s))
        self.assertEqual(self.project_id, a.get_project_id(s))

    def assertAccessInfoEqual(self, a, b):
        self.assertEqual(a.auth_token, b.auth_token)
        self.assertEqual(a._data, b._data)

    def test_check_cache_id_match(self):
        a = self.create_auth_plugin()
        b = self.create_auth_plugin()
        self.assertIsNot(a, b)
        self.assertIsNone(a.get_auth_state())
        self.assertIsNone(b.get_auth_state())
        a_id = a.get_cache_id()
        b_id = b.get_cache_id()
        self.assertIsNotNone(a_id)
        self.assertIsNotNone(b_id)
        self.assertEqual(a_id, b_id)

    def test_check_cache_id_no_match(self):
        a = self.create_auth_plugin(project_id='a')
        b = self.create_auth_plugin(project_id='b')
        self.assertIsNot(a, b)
        self.assertIsNone(a.get_auth_state())
        self.assertIsNone(b.get_auth_state())
        a_id = a.get_cache_id()
        b_id = b.get_cache_id()
        self.assertIsNotNone(a_id)
        self.assertIsNotNone(b_id)
        self.assertNotEqual(a_id, b_id)

    def test_get_set_auth_state(self):
        a = self.create_auth_plugin()
        b = self.create_auth_plugin()
        self.assertEqual(a.get_cache_id(), b.get_cache_id())
        s = session.Session()
        a_token = a.get_token(s)
        self.assertEqual(1, self.requests_mock.call_count)
        auth_state = a.get_auth_state()
        self.assertIsNotNone(auth_state)
        b.set_auth_state(auth_state)
        b_token = b.get_token(s)
        self.assertEqual(1, self.requests_mock.call_count)
        self.assertEqual(a_token, b_token)
        self.assertAccessInfoEqual(a.auth_ref, b.auth_ref)

    def test_pathless_url(self):
        disc = fixture.DiscoveryList(v2=False, v3=False)
        url = 'http://path.less.url:1234'
        disc.add_microversion(href=url, id='v2.1')
        self.stub_url('GET', base_url=url, status_code=200, json=disc)
        token = fixture.V2Token()
        service = token.add_service('network')
        service.add_endpoint(public=url, admin=url, internal=url)
        self.stub_url('POST', ['tokens'], base_url=url, json=token)
        v2_auth = identity.V2Password(url, username='u', password='p')
        sess = session.Session(auth=v2_auth)
        data = sess.get_endpoint_data(service_type='network')
        self.assertEqual(url, data.url)
        self.assertEqual((2, 1), data.api_version)
        self.assertEqual(3, len(list(data._get_discovery_url_choices(project_id='42'))))