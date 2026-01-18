import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
class TestObjectController(testtools.TestCase):

    def setUp(self):
        super(TestObjectController, self).setUp()
        self.api = utils.FakeAPI(data_fixtures)
        self.schema_api = utils.FakeSchemaAPI(schema_fixtures)
        self.controller = base.BaseController(self.api, self.schema_api, metadefs.ObjectController)

    def test_list_object(self):
        objects = self.controller.list(NAMESPACE1)
        actual = [obj.name for obj in objects]
        self.assertEqual([OBJECT1, OBJECT2], actual)

    def test_get_object(self):
        obj = self.controller.get(NAMESPACE1, OBJECT1)
        self.assertEqual(OBJECT1, obj.name)
        self.assertEqual(sorted([PROPERTY1, PROPERTY2]), sorted(list(obj.properties.keys())))

    def test_create_object(self):
        properties = {'name': OBJECTNEW, 'description': 'DESCRIPTION'}
        obj = self.controller.create(NAMESPACE1, **properties)
        self.assertEqual(OBJECTNEW, obj.name)

    def test_create_object_invalid_property(self):
        properties = {'namespace': NAMESPACE1}
        self.assertRaises(TypeError, self.controller.create, **properties)

    def test_update_object(self):
        properties = {'description': 'UPDATED_DESCRIPTION'}
        obj = self.controller.update(NAMESPACE1, OBJECT1, **properties)
        self.assertEqual(OBJECT1, obj.name)

    def test_update_object_invalid_property(self):
        properties = {'required': 'INVALID'}
        self.assertRaises(TypeError, self.controller.update, NAMESPACE1, OBJECT1, **properties)

    def test_update_object_disallowed_fields(self):
        properties = {'description': 'UPDATED_DESCRIPTION'}
        self.controller.update(NAMESPACE1, OBJECT1, **properties)
        actual = self.api.calls
        "('PUT', '/v2/metadefs/namespaces/Namespace1/objects/Object1', {},\n        [('description', 'UPDATED_DESCRIPTION'),\n        ('name', 'Object1'),\n        ('properties', ...),\n        ('required', [])])"
        _disallowed_fields = ['self', 'schema', 'created_at', 'updated_at']
        for key in actual[1][3]:
            self.assertNotIn(key, _disallowed_fields)

    def test_delete_object(self):
        self.controller.delete(NAMESPACE1, OBJECT1)
        expect = [('DELETE', '/v2/metadefs/namespaces/%s/objects/%s' % (NAMESPACE1, OBJECT1), {}, None)]
        self.assertEqual(expect, self.api.calls)

    def test_delete_all_objects(self):
        self.controller.delete_all(NAMESPACE1)
        expect = [('DELETE', '/v2/metadefs/namespaces/%s/objects' % NAMESPACE1, {}, None)]
        self.assertEqual(expect, self.api.calls)