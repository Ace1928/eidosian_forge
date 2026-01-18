import http.client as http
from oslo_serialization import jsonutils
import webob
from glance.common import auth
from glance.common import exception
from glance.tests import utils
class TestEndpoints(utils.BaseTestCase):

    def setUp(self):
        super(TestEndpoints, self).setUp()
        self.service_catalog = [{'endpoint_links': [], 'endpoints': [{'adminURL': 'http://localhost:8080/', 'region': 'RegionOne', 'internalURL': 'http://internalURL/', 'publicURL': 'http://publicURL/'}], 'type': 'object-store', 'name': 'Object Storage Service'}]

    def test_get_endpoint_with_custom_server_type(self):
        endpoint = auth.get_endpoint(self.service_catalog, service_type='object-store')
        self.assertEqual('http://publicURL/', endpoint)

    def test_get_endpoint_with_custom_endpoint_type(self):
        endpoint = auth.get_endpoint(self.service_catalog, service_type='object-store', endpoint_type='internalURL')
        self.assertEqual('http://internalURL/', endpoint)

    def test_get_endpoint_raises_with_invalid_service_type(self):
        self.assertRaises(exception.NoServiceEndpoint, auth.get_endpoint, self.service_catalog, service_type='foo')

    def test_get_endpoint_raises_with_invalid_endpoint_type(self):
        self.assertRaises(exception.NoServiceEndpoint, auth.get_endpoint, self.service_catalog, service_type='object-store', endpoint_type='foo')

    def test_get_endpoint_raises_with_invalid_endpoint_region(self):
        self.assertRaises(exception.NoServiceEndpoint, auth.get_endpoint, self.service_catalog, service_type='object-store', endpoint_region='foo', endpoint_type='internalURL')