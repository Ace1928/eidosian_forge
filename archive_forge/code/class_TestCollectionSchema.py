from glance.common import exception
import glance.schema
from glance.tests import utils as test_utils
class TestCollectionSchema(test_utils.BaseTestCase):

    def test_raw_json_schema(self):
        item_properties = {'cheese': {'type': 'string'}}
        item_schema = glance.schema.Schema('mouse', item_properties)
        collection_schema = glance.schema.CollectionSchema('mice', item_schema)
        expected = {'name': 'mice', 'properties': {'mice': {'type': 'array', 'items': item_schema.raw()}, 'first': {'type': 'string'}, 'next': {'type': 'string'}, 'schema': {'type': 'string'}}, 'links': [{'rel': 'first', 'href': '{first}'}, {'rel': 'next', 'href': '{next}'}, {'rel': 'describedby', 'href': '{schema}'}]}
        self.assertEqual(expected, collection_schema.raw())