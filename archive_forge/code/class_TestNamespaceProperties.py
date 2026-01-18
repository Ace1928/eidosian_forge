import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
class TestNamespaceProperties(metadef_base.MetadefFunctionalTestBase):

    def setUp(self):
        super(TestNamespaceProperties, self).setUp()
        self.cleanup()
        self.api_server.deployment_flavor = 'noauth'
        self.start_servers(**self.__dict__.copy())

    def test_properties_lifecycle(self):
        path = self._url('/v2/metadefs/namespaces/MyNamespace')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/metadefs/namespaces')
        headers = self._headers({'content-type': 'application/json'})
        namespace_name = 'MyNamespace'
        resource_type_name = 'MyResourceType'
        resource_type_prefix = 'MyPrefix'
        data = jsonutils.dumps({'namespace': namespace_name, 'display_name': 'My User Friendly Namespace', 'description': 'My description', 'visibility': 'public', 'protected': False, 'owner': 'The Test Owner', 'resource_type_associations': [{'name': resource_type_name, 'prefix': resource_type_prefix}]})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        path = self._url('/v2/metadefs/namespaces/MyNamespace/properties/property1')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/metadefs/namespaces/MyNamespace/properties')
        headers = self._headers({'content-type': 'application/json'})
        property_name = 'property1'
        data = jsonutils.dumps({'name': property_name, 'type': 'integer', 'title': 'property1', 'description': 'property1 description', 'default': 100, 'minimum': 100, 'maximum': 30000369, 'readonly': False})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CONFLICT, response.status_code)
        path = self._url('/v2/metadefs/namespaces/%s/properties/%s' % (namespace_name, property_name))
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        property_object = jsonutils.loads(response.text)
        self.assertEqual('integer', property_object['type'])
        self.assertEqual('property1', property_object['title'])
        self.assertEqual('property1 description', property_object['description'])
        self.assertEqual('100', property_object['default'])
        self.assertEqual(100, property_object['minimum'])
        self.assertEqual(30000369, property_object['maximum'])
        path = self._url('/v2/metadefs/namespaces/%s/properties/%s%s' % (namespace_name, property_name, '='.join(['?resource_type', resource_type_name])))
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        property_name_with_prefix = ''.join([resource_type_prefix, property_name])
        path = self._url('/v2/metadefs/namespaces/%s/properties/%s%s' % (namespace_name, property_name_with_prefix, '='.join(['?resource_type', resource_type_name])))
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        property_object = jsonutils.loads(response.text)
        self.assertEqual('integer', property_object['type'])
        self.assertEqual('property1', property_object['title'])
        self.assertEqual('property1 description', property_object['description'])
        self.assertEqual('100', property_object['default'])
        self.assertEqual(100, property_object['minimum'])
        self.assertEqual(30000369, property_object['maximum'])
        self.assertFalse(property_object['readonly'])
        property_object = jsonutils.loads(response.text)
        checked_keys = set(['name', 'type', 'title', 'description', 'default', 'minimum', 'maximum', 'readonly'])
        self.assertEqual(set(property_object.keys()), checked_keys)
        expected_metadata_property = {'type': 'integer', 'title': 'property1', 'description': 'property1 description', 'default': '100', 'minimum': 100, 'maximum': 30000369, 'readonly': False}
        for key, value in expected_metadata_property.items():
            self.assertEqual(property_object[key], value, key)
        path = self._url('/v2/metadefs/namespaces/%s/properties/%s' % (namespace_name, property_name))
        media_type = 'application/json'
        headers = self._headers({'content-type': media_type})
        property_name = 'property1-UPDATED'
        data = jsonutils.dumps({'name': property_name, 'type': 'string', 'title': 'string property', 'description': 'desc-UPDATED', 'operators': ['<or>'], 'default': 'value-UPDATED', 'minLength': 5, 'maxLength': 10, 'readonly': True})
        response = requests.put(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        property_object = jsonutils.loads(response.text)
        self.assertEqual('string', property_object['type'])
        self.assertEqual('desc-UPDATED', property_object['description'])
        self.assertEqual('value-UPDATED', property_object['default'])
        self.assertEqual(['<or>'], property_object['operators'])
        self.assertEqual(5, property_object['minLength'])
        self.assertEqual(10, property_object['maxLength'])
        self.assertTrue(property_object['readonly'])
        path = self._url('/v2/metadefs/namespaces/%s/properties/%s' % (namespace_name, property_name))
        response = requests.get(path, headers=self._headers())
        self.assertEqual('string', property_object['type'])
        self.assertEqual('desc-UPDATED', property_object['description'])
        self.assertEqual('value-UPDATED', property_object['default'])
        self.assertEqual(['<or>'], property_object['operators'])
        self.assertEqual(5, property_object['minLength'])
        self.assertEqual(10, property_object['maxLength'])
        path = self._url('/v2/metadefs/namespaces/%s/properties/%s' % (namespace_name, property_name))
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/metadefs/namespaces/%s/properties/%s' % (namespace_name, property_name))
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)

    def _create_properties(self, namespaces):
        properties = []
        for namespace in namespaces:
            headers = self._headers({'X-Tenant-Id': namespace['owner']})
            data = {'name': 'property_of_%s' % namespace['namespace'], 'type': 'integer', 'title': 'property', 'description': 'property description'}
            path = self._url('/v2/metadefs/namespaces/%s/properties' % namespace['namespace'])
            response = requests.post(path, headers=headers, json=data)
            self.assertEqual(http.CREATED, response.status_code)
            prop_metadata = response.json()
            metadef_property = dict()
            metadef_property[namespace['namespace']] = prop_metadata['name']
            properties.append(metadef_property)
        return properties

    def _update_property(self, path, headers, data):
        response = requests.put(path, headers=headers, json=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        property_object = response.json()
        self.assertEqual('string', property_object['type'])
        self.assertEqual(data['description'], property_object['description'])
        response = requests.get(path, headers=self._headers())
        self.assertEqual('string', property_object['type'])
        self.assertEqual(data['description'], property_object['description'])

    def test_role_base_metadata_properties_lifecycle(self):
        path = self._url('/v2/metadefs/namespaces')
        headers = self._headers({'content-type': 'application/json'})
        tenant1_namespaces = []
        tenant2_namespaces = []
        for tenant in [self.tenant1, self.tenant2]:
            headers['X-Tenant-Id'] = tenant
            for visibility in ['public', 'private']:
                namespace_data = {'namespace': '%s_%s_namespace' % (tenant, visibility), 'display_name': 'My User Friendly Namespace', 'description': 'My description', 'visibility': visibility, 'owner': tenant}
                namespace = self.create_namespace(path, headers, namespace_data)
                self.assertNamespacesEqual(namespace, namespace_data)
                if tenant == self.tenant1:
                    tenant1_namespaces.append(namespace)
                else:
                    tenant2_namespaces.append(namespace)
        tenant1_properties = self._create_properties(tenant1_namespaces)
        tenant2_properties = self._create_properties(tenant2_namespaces)

        def _check_properties_access(properties, tenant):
            headers = self._headers({'content-type': 'application/json', 'X-Tenant-Id': tenant, 'X-Roles': 'reader,member'})
            for prop in properties:
                for namespace, property_name in prop.items():
                    path = self._url('/v2/metadefs/namespaces/%s/properties/%s' % (namespace, property_name))
                    response = requests.get(path, headers=headers)
                    if namespace.split('_')[1] == 'public':
                        expected = http.OK
                    else:
                        expected = http.NOT_FOUND
                    self.assertEqual(expected, response.status_code)
                    path = self._url('/v2/metadefs/namespaces/%s/properties' % namespace)
                    response = requests.get(path, headers=headers)
                    self.assertEqual(expected, response.status_code)
                    if expected == http.OK:
                        resp_props = response.json()['properties'].values()
                        self.assertEqual(sorted(prop.values()), sorted([x['name'] for x in resp_props]))
        _check_properties_access(tenant2_properties, self.tenant1)
        _check_properties_access(tenant1_properties, self.tenant2)
        total_properties = tenant1_properties + tenant2_properties
        for prop in total_properties:
            for namespace, property_name in prop.items():
                data = {'name': property_name, 'type': 'string', 'title': 'string property', 'description': 'desc-UPDATED'}
                path = self._url('/v2/metadefs/namespaces/%s/properties/%s' % (namespace, property_name))
                headers['X-Roles'] = 'reader,member'
                response = requests.put(path, headers=headers, json=data)
                self.assertEqual(http.FORBIDDEN, response.status_code)
                headers = self._headers({'X-Tenant-Id': namespace.split('_')[0]})
                self._update_property(path, headers, data)
        for prop in total_properties:
            for namespace, property_name in prop.items():
                path = self._url('/v2/metadefs/namespaces/%s/properties/%s' % (namespace, property_name))
                response = requests.delete(path, headers=self._headers({'X-Roles': 'reader,member', 'X-Tenant-Id': namespace.split('_')[0]}))
                self.assertEqual(http.FORBIDDEN, response.status_code)
        headers = self._headers()
        for prop in total_properties:
            for namespace, property_name in prop.items():
                path = self._url('/v2/metadefs/namespaces/%s/properties/%s' % (namespace, property_name))
                response = requests.delete(path, headers=headers)
                self.assertEqual(http.NO_CONTENT, response.status_code)
                response = requests.get(path, headers=headers)
                self.assertEqual(http.NOT_FOUND, response.status_code)