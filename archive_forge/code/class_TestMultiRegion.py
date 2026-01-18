import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
class TestMultiRegion(test_v3.RestfulTestCase):

    def test_catalog_with_multi_region_reports_all_endpoints(self):
        first_region = self.post('/regions', body={'region': unit.new_region_ref()}).json_body['region']
        second_region = self.post('/regions', body={'region': unit.new_region_ref()}).json_body['region']
        first_service = self.post('/services', body={'service': unit.new_service_ref(type='foobar')}).json_body['service']
        second_service = self.post('/services', body={'service': unit.new_service_ref(type='foobar')}).json_body['service']
        first_endpoint = self.post('/endpoints', body={'endpoint': unit.new_endpoint_ref(first_service['id'], interface='public', region_id=first_region['id'])}).json_body['endpoint']
        second_endpoint = self.post('/endpoints', body={'endpoint': unit.new_endpoint_ref(second_service['id'], interface='public', region_id=second_region['id'])}).json_body['endpoint']
        found_first_endpoint = False
        found_second_endpoint = False
        catalog = self.get('/auth/catalog/').json_body['catalog']
        for service in catalog:
            if service['id'] == first_service['id']:
                endpoint = service['endpoints'][0]
                self.assertEqual(endpoint['id'], first_endpoint['id'])
                self.assertEqual(endpoint['region_id'], first_region['id'])
                found_first_endpoint = True
            elif service['id'] == second_service['id']:
                endpoint = service['endpoints'][0]
                self.assertEqual(endpoint['id'], second_endpoint['id'])
                self.assertEqual(endpoint['region_id'], second_region['id'])
                found_second_endpoint = True
        self.assertTrue(found_first_endpoint)
        self.assertTrue(found_second_endpoint)