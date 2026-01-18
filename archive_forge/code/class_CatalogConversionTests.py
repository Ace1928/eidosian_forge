import itertools
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystonemiddleware.auth_token import _request
from keystonemiddleware.tests.unit import utils
class CatalogConversionTests(utils.TestCase):
    PUBLIC_URL = 'http://server:5000/v2.0'
    ADMIN_URL = 'http://admin:35357/v2.0'
    INTERNAL_URL = 'http://internal:5000/v2.0'
    REGION_ONE = 'RegionOne'
    REGION_TWO = 'RegionTwo'
    REGION_THREE = 'RegionThree'

    def test_basic_convert(self):
        token = fixture.V3Token()
        s = token.add_service(type='identity')
        s.add_standard_endpoints(public=self.PUBLIC_URL, admin=self.ADMIN_URL, internal=self.INTERNAL_URL, region=self.REGION_ONE)
        auth_ref = access.create(body=token)
        catalog_data = auth_ref.service_catalog.catalog
        catalog = _request._normalize_catalog(catalog_data)
        self.assertEqual(1, len(catalog))
        service = catalog[0]
        self.assertEqual(1, len(service['endpoints']))
        endpoints = service['endpoints'][0]
        self.assertEqual('identity', service['type'])
        self.assertEqual(4, len(endpoints))
        self.assertEqual(self.PUBLIC_URL, endpoints['publicURL'])
        self.assertEqual(self.ADMIN_URL, endpoints['adminURL'])
        self.assertEqual(self.INTERNAL_URL, endpoints['internalURL'])
        self.assertEqual(self.REGION_ONE, endpoints['region'])

    def test_multi_region(self):
        token = fixture.V3Token()
        s = token.add_service(type='identity')
        s.add_endpoint('internal', self.INTERNAL_URL, region=self.REGION_ONE)
        s.add_endpoint('public', self.PUBLIC_URL, region=self.REGION_TWO)
        s.add_endpoint('admin', self.ADMIN_URL, region=self.REGION_THREE)
        auth_ref = access.create(body=token)
        catalog_data = auth_ref.service_catalog.catalog
        catalog = _request._normalize_catalog(catalog_data)
        self.assertEqual(1, len(catalog))
        service = catalog[0]
        expected = [{'internalURL': self.INTERNAL_URL, 'region': self.REGION_ONE}, {'publicURL': self.PUBLIC_URL, 'region': self.REGION_TWO}, {'adminURL': self.ADMIN_URL, 'region': self.REGION_THREE}]
        self.assertEqual('identity', service['type'])
        self.assertEqual(3, len(service['endpoints']))
        for e in expected:
            self.assertIn(e, expected)