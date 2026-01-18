import http.client as http_client
import httplib2
from oslo_serialization import jsonutils
from glance.tests import functional
from glance.tests.unit import test_versions as tv
class TestApiPaths(functional.FunctionalTest):

    def setUp(self):
        super(TestApiPaths, self).setUp()
        self.start_servers(**self.__dict__.copy())
        url = 'http://127.0.0.1:%d' % self.api_port
        self.versions = {'versions': tv.get_versions_list(url, enabled_cache=True)}
        images = {'images': []}
        self.images_json = jsonutils.dumps(images)

    def test_get_root_path(self):
        """Assert GET / with `no Accept:` header.
        Verify version choices returned.
        Bug lp:803260  no Accept header causes a 500 in glance-api
        """
        path = 'http://%s:%d' % ('127.0.0.1', self.api_port)
        http = httplib2.Http()
        response, content_json = http.request(path, 'GET')
        self.assertEqual(http_client.MULTIPLE_CHOICES, response.status)
        content = jsonutils.loads(content_json.decode())
        self.assertEqual(self.versions, content)

    def test_get_root_path_with_unknown_header(self):
        """Assert GET / with Accept: unknown header
        Verify version choices returned. Verify message in API log about
        unknown accept header.
        """
        path = 'http://%s:%d/' % ('127.0.0.1', self.api_port)
        http = httplib2.Http()
        headers = {'Accept': 'unknown'}
        response, content_json = http.request(path, 'GET', headers=headers)
        self.assertEqual(http_client.MULTIPLE_CHOICES, response.status)
        content = jsonutils.loads(content_json.decode())
        self.assertEqual(self.versions, content)

    def test_get_va1_images_path(self):
        """Assert GET /va.1/images with no Accept: header
        Verify version choices returned
        """
        path = 'http://%s:%d/va.1/images' % ('127.0.0.1', self.api_port)
        http = httplib2.Http()
        response, content_json = http.request(path, 'GET')
        self.assertEqual(http_client.MULTIPLE_CHOICES, response.status)
        content = jsonutils.loads(content_json.decode())
        self.assertEqual(self.versions, content)

    def test_get_versions_path(self):
        """Assert GET /versions with no Accept: header
        Verify version choices returned
        """
        path = 'http://%s:%d/versions' % ('127.0.0.1', self.api_port)
        http = httplib2.Http()
        response, content_json = http.request(path, 'GET')
        self.assertEqual(http_client.OK, response.status)
        content = jsonutils.loads(content_json.decode())
        self.assertEqual(self.versions, content)

    def test_get_versions_choices(self):
        """Verify version choices returned"""
        path = 'http://%s:%d/v10' % ('127.0.0.1', self.api_port)
        http = httplib2.Http()
        response, content_json = http.request(path, 'GET')
        self.assertEqual(http_client.MULTIPLE_CHOICES, response.status)
        content = jsonutils.loads(content_json.decode())
        self.assertEqual(self.versions, content)