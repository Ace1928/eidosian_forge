import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
class ServiceCatalogV3Test(ServiceCatalogTest):

    def test_building_a_service_catalog(self):
        sc = access.create(auth_token=uuid.uuid4().hex, body=self.AUTH_RESPONSE_BODY).service_catalog
        self.assertEqual(sc.url_for(service_type='compute'), 'https://compute.north.host/novapi/public')
        self.assertEqual(sc.url_for(service_type='compute', interface='internal'), 'https://compute.north.host/novapi/internal')
        self.assertRaises(exceptions.EndpointNotFound, sc.url_for, region_name='South', service_type='compute')

    def test_service_catalog_endpoints(self):
        sc = access.create(auth_token=uuid.uuid4().hex, body=self.AUTH_RESPONSE_BODY).service_catalog
        public_ep = sc.get_endpoints(service_type='compute', interface='public')
        self.assertEqual(public_ep['compute'][0]['region_id'], 'North')
        self.assertEqual(public_ep['compute'][0]['url'], 'https://compute.north.host/novapi/public')

    def test_service_catalog_multiple_service_types(self):
        token = fixture.V3Token()
        token.set_project_scope()
        for i in range(3):
            s = token.add_service('compute')
            s.add_standard_endpoints(public='public-%d' % i, admin='admin-%d' % i, internal='internal-%d' % i, region='region-%d' % i)
        auth_ref = access.create(resp=None, body=token)
        urls = auth_ref.service_catalog.get_urls(service_type='compute', interface='public')
        self.assertEqual(set(['public-0', 'public-1', 'public-2']), set(urls))
        urls = auth_ref.service_catalog.get_urls(service_type='compute', interface='public', region_name='region-1')
        self.assertEqual(('public-1',), urls)

    def test_service_catalog_endpoint_id(self):
        token = fixture.V3Token()
        token.set_project_scope()
        service_id = uuid.uuid4().hex
        endpoint_id = uuid.uuid4().hex
        public_url = uuid.uuid4().hex
        s = token.add_service('compute', id=service_id)
        s.add_endpoint('public', public_url, id=endpoint_id)
        s.add_endpoint('public', uuid.uuid4().hex)
        auth_ref = access.create(body=token)
        urls = auth_ref.service_catalog.get_urls(service_type='compute', interface='public')
        self.assertEqual(2, len(urls))
        urls = auth_ref.service_catalog.get_urls(service_type='compute', endpoint_id=uuid.uuid4().hex, interface='public')
        self.assertEqual(0, len(urls))
        urls = auth_ref.service_catalog.get_urls(service_type='compute', service_id=service_id, interface='public')
        self.assertEqual(2, len(urls))
        urls = auth_ref.service_catalog.get_urls(service_type='compute', service_id=service_id, endpoint_id=endpoint_id, interface='public')
        self.assertEqual((public_url,), urls)
        urls = auth_ref.service_catalog.get_urls(service_type='compute', endpoint_id=endpoint_id, interface='public')
        self.assertEqual((public_url,), urls)

    def test_service_catalog_without_service_type(self):
        token = fixture.V3Token()
        token.set_project_scope()
        public_urls = []
        for i in range(0, 3):
            public_url = uuid.uuid4().hex
            public_urls.append(public_url)
            s = token.add_service(uuid.uuid4().hex)
            s.add_endpoint('public', public_url)
        auth_ref = access.create(body=token)
        urls = auth_ref.service_catalog.get_urls(interface='public')
        self.assertEqual(3, len(urls))
        for p in public_urls:
            self.assertIn(p, urls)