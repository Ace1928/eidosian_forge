from glance.common import exception
import glance.schema
from glance.tests import utils as test_utils
class TestBasicSchema(test_utils.BaseTestCase):

    def setUp(self):
        super(TestBasicSchema, self).setUp()
        properties = {'ham': {'type': 'string'}, 'eggs': {'type': 'string'}}
        self.schema = glance.schema.Schema('basic', properties)

    def test_validate_passes(self):
        obj = {'ham': 'no', 'eggs': 'scrambled'}
        self.schema.validate(obj)

    def test_validate_fails_on_extra_properties(self):
        obj = {'ham': 'virginia', 'eggs': 'scrambled', 'bacon': 'crispy'}
        self.assertRaises(exception.InvalidObject, self.schema.validate, obj)

    def test_validate_fails_on_bad_type(self):
        obj = {'eggs': 2}
        self.assertRaises(exception.InvalidObject, self.schema.validate, obj)

    def test_filter_strips_extra_properties(self):
        obj = {'ham': 'virginia', 'eggs': 'scrambled', 'bacon': 'crispy'}
        filtered = self.schema.filter(obj)
        expected = {'ham': 'virginia', 'eggs': 'scrambled'}
        self.assertEqual(expected, filtered)

    def test_merge_properties(self):
        self.schema.merge_properties({'bacon': {'type': 'string'}})
        expected = set(['ham', 'eggs', 'bacon'])
        actual = set(self.schema.raw()['properties'].keys())
        self.assertEqual(expected, actual)

    def test_merge_conflicting_properties(self):
        conflicts = {'eggs': {'type': 'integer'}}
        self.assertRaises(exception.SchemaLoadError, self.schema.merge_properties, conflicts)

    def test_merge_conflicting_but_identical_properties(self):
        conflicts = {'ham': {'type': 'string'}}
        self.schema.merge_properties(conflicts)
        expected = set(['ham', 'eggs'])
        actual = set(self.schema.raw()['properties'].keys())
        self.assertEqual(expected, actual)

    def test_raw_json_schema(self):
        expected = {'name': 'basic', 'properties': {'ham': {'type': 'string'}, 'eggs': {'type': 'string'}}, 'additionalProperties': False}
        self.assertEqual(expected, self.schema.raw())