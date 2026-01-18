import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
class TestNamespaceController(testtools.TestCase):

    def setUp(self):
        super(TestNamespaceController, self).setUp()
        self.api = utils.FakeAPI(data_fixtures)
        self.schema_api = utils.FakeSchemaAPI(schema_fixtures)
        self.controller = base.BaseController(self.api, self.schema_api, metadefs.NamespaceController)

    def test_list_namespaces(self):
        namespaces = self.controller.list()
        self.assertEqual(2, len(namespaces))
        self.assertEqual(NAMESPACE1, namespaces[0]['namespace'])
        self.assertEqual(NAMESPACE2, namespaces[1]['namespace'])

    def test_list_namespaces_paginate(self):
        namespaces = self.controller.list(page_size=1)
        self.assertEqual(2, len(namespaces))
        self.assertEqual(NAMESPACE7, namespaces[0]['namespace'])
        self.assertEqual(NAMESPACE8, namespaces[1]['namespace'])

    def test_list_with_limit_greater_than_page_size(self):
        namespaces = self.controller.list(page_size=1, limit=2)
        self.assertEqual(2, len(namespaces))
        self.assertEqual(NAMESPACE7, namespaces[0]['namespace'])
        self.assertEqual(NAMESPACE8, namespaces[1]['namespace'])

    def test_list_with_marker(self):
        namespaces = self.controller.list(marker=NAMESPACE6, page_size=2)
        self.assertEqual(2, len(namespaces))
        self.assertEqual(NAMESPACE7, namespaces[0]['namespace'])
        self.assertEqual(NAMESPACE8, namespaces[1]['namespace'])

    def test_list_with_sort_dir(self):
        namespaces = self.controller.list(sort_dir='asc', limit=1)
        self.assertEqual(1, len(namespaces))
        self.assertEqual(NAMESPACE1, namespaces[0]['namespace'])

    def test_list_with_sort_dir_invalid(self):
        self.assertRaises(ValueError, self.controller.list, sort_dir='foo')

    def test_list_with_sort_key(self):
        namespaces = self.controller.list(sort_key='created_at', limit=1)
        self.assertEqual(1, len(namespaces))
        self.assertEqual(NAMESPACE1, namespaces[0]['namespace'])

    def test_list_with_sort_key_invalid(self):
        self.assertRaises(ValueError, self.controller.list, sort_key='foo')

    def test_list_namespaces_with_one_resource_type_filter(self):
        namespaces = self.controller.list(filters={'resource_types': [RESOURCE_TYPE1]})
        self.assertEqual(1, len(namespaces))
        self.assertEqual(NAMESPACE3, namespaces[0]['namespace'])

    def test_list_namespaces_with_multiple_resource_types_filter(self):
        namespaces = self.controller.list(filters={'resource_types': [RESOURCE_TYPE1, RESOURCE_TYPE2]})
        self.assertEqual(1, len(namespaces))
        self.assertEqual(NAMESPACE4, namespaces[0]['namespace'])

    def test_list_namespaces_with_visibility_filter(self):
        namespaces = self.controller.list(filters={'visibility': 'private'})
        self.assertEqual(1, len(namespaces))
        self.assertEqual(NAMESPACE5, namespaces[0]['namespace'])

    def test_get_namespace(self):
        namespace = self.controller.get(NAMESPACE1)
        self.assertEqual(NAMESPACE1, namespace.namespace)
        self.assertTrue(namespace.protected)

    def test_get_namespace_with_resource_type(self):
        namespace = self.controller.get(NAMESPACE6, resource_type=RESOURCE_TYPE1)
        self.assertEqual(NAMESPACE6, namespace.namespace)
        self.assertTrue(namespace.protected)

    def test_create_namespace(self):
        properties = {'namespace': NAMESPACENEW}
        namespace = self.controller.create(**properties)
        self.assertEqual(NAMESPACENEW, namespace.namespace)
        self.assertTrue(namespace.protected)

    def test_create_namespace_invalid_data(self):
        properties = {}
        self.assertRaises(TypeError, self.controller.create, **properties)

    def test_create_namespace_invalid_property(self):
        properties = {'namespace': 'NewNamespace', 'protected': '123'}
        self.assertRaises(TypeError, self.controller.create, **properties)

    def test_update_namespace(self):
        properties = {'display_name': 'My Updated Name'}
        namespace = self.controller.update(NAMESPACE1, **properties)
        self.assertEqual(NAMESPACE1, namespace.namespace)

    def test_update_namespace_invalid_property(self):
        properties = {'protected': '123'}
        self.assertRaises(TypeError, self.controller.update, NAMESPACE1, **properties)

    def test_update_namespace_disallowed_fields(self):
        properties = {'display_name': 'My Updated Name'}
        self.controller.update(NAMESPACE1, **properties)
        actual = self.api.calls
        _disallowed_fields = ['self', 'schema', 'created_at', 'updated_at']
        for key in actual[1][3]:
            self.assertNotIn(key, _disallowed_fields)

    def test_delete_namespace(self):
        self.controller.delete(NAMESPACE1)
        expect = [('DELETE', '/v2/metadefs/namespaces/%s' % NAMESPACE1, {}, None)]
        self.assertEqual(expect, self.api.calls)