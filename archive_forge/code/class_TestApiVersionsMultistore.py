import http.client as http_client
import httplib2
from oslo_serialization import jsonutils
from glance.tests import functional
from glance.tests.unit import test_versions as tv
class TestApiVersionsMultistore(functional.MultipleBackendFunctionalTest):

    def test_version_configurations(self):
        """Test that versioning is handled properly through all channels"""
        self.start_servers(**self.__dict__.copy())
        url = 'http://127.0.0.1:%d' % self.api_port
        versions = {'versions': tv.get_versions_list(url, enabled_backends=True, enabled_cache=True)}
        path = 'http://%s:%d' % ('127.0.0.1', self.api_port)
        http = httplib2.Http()
        response, content_json = http.request(path, 'GET')
        self.assertEqual(http_client.MULTIPLE_CHOICES, response.status)
        content = jsonutils.loads(content_json.decode())
        self.assertEqual(versions, content)

    def test_v2_api_configuration(self):
        self.start_servers(**self.__dict__.copy())
        url = 'http://127.0.0.1:%d' % self.api_port
        versions = {'versions': tv.get_versions_list(url, enabled_backends=True, enabled_cache=True)}
        path = 'http://%s:%d' % ('127.0.0.1', self.api_port)
        http = httplib2.Http()
        response, content_json = http.request(path, 'GET')
        self.assertEqual(http_client.MULTIPLE_CHOICES, response.status)
        content = jsonutils.loads(content_json.decode())
        self.assertEqual(versions, content)