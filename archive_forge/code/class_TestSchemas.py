import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests import functional
class TestSchemas(functional.FunctionalTest):

    def setUp(self):
        super(TestSchemas, self).setUp()
        self.cleanup()
        self.start_servers(**self.__dict__.copy())

    def test_resource(self):
        path = 'http://%s:%d/v2/schemas/image' % ('127.0.0.1', self.api_port)
        response = requests.get(path)
        self.assertEqual(http.OK, response.status_code)
        image_schema = jsonutils.loads(response.text)
        expected = set(['id', 'name', 'visibility', 'checksum', 'os_hash_algo', 'os_hash_value', 'created_at', 'updated_at', 'tags', 'size', 'virtual_size', 'owner', 'container_format', 'disk_format', 'self', 'file', 'status', 'schema', 'direct_url', 'locations', 'min_ram', 'min_disk', 'protected', 'os_hidden', 'stores'])
        self.assertEqual(expected, set(image_schema['properties'].keys()))
        path = 'http://%s:%d/v2/schemas/images' % ('127.0.0.1', self.api_port)
        response = requests.get(path)
        self.assertEqual(http.OK, response.status_code)
        images_schema = jsonutils.loads(response.text)
        item_schema = images_schema['properties']['images']['items']
        self.assertEqual(item_schema, image_schema)
        self.stop_servers()