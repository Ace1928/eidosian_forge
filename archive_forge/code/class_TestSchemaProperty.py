import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
class TestSchemaProperty(testtools.TestCase):

    def test_property_minimum(self):
        prop = schemas.SchemaProperty('size')
        self.assertEqual('size', prop.name)

    def test_property_description(self):
        prop = schemas.SchemaProperty('size', description='some quantity')
        self.assertEqual('size', prop.name)
        self.assertEqual('some quantity', prop.description)

    def test_property_is_base(self):
        prop1 = schemas.SchemaProperty('name')
        prop2 = schemas.SchemaProperty('foo', is_base=False)
        prop3 = schemas.SchemaProperty('foo', is_base=True)
        self.assertTrue(prop1.is_base)
        self.assertFalse(prop2.is_base)
        self.assertTrue(prop3.is_base)