from glance.common import exception
import glance.schema
from glance.tests import utils as test_utils
class TestBasicSchemaLinks(test_utils.BaseTestCase):

    def setUp(self):
        super(TestBasicSchemaLinks, self).setUp()
        properties = {'ham': {'type': 'string'}, 'eggs': {'type': 'string'}}
        links = [{'rel': 'up', 'href': '/menu'}]
        self.schema = glance.schema.Schema('basic', properties, links)

    def test_raw_json_schema(self):
        expected = {'name': 'basic', 'properties': {'ham': {'type': 'string'}, 'eggs': {'type': 'string'}}, 'links': [{'rel': 'up', 'href': '/menu'}], 'additionalProperties': False}
        self.assertEqual(expected, self.schema.raw())