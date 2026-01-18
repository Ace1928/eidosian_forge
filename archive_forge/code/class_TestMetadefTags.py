import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
class TestMetadefTags(metadef_base.MetadefFunctionalTestBase):

    def setUp(self):
        super(TestMetadefTags, self).setUp()
        self.cleanup()
        self.api_server.deployment_flavor = 'noauth'
        self.start_servers(**self.__dict__.copy())

    def test_metadata_tags_lifecycle(self):
        path = self._url('/v2/metadefs/namespaces/MyNamespace')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/metadefs/namespaces')
        headers = self._headers({'content-type': 'application/json'})
        namespace_name = 'MyNamespace'
        data = jsonutils.dumps({'namespace': namespace_name, 'display_name': 'My User Friendly Namespace', 'description': 'My description', 'visibility': 'public', 'protected': False, 'owner': 'The Test Owner'})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        metadata_tag_name = 'tag1'
        path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace_name, metadata_tag_name))
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        headers = self._headers({'content-type': 'application/json'})
        response = requests.post(path, headers=headers)
        self.assertEqual(http.CREATED, response.status_code)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        metadata_tag = jsonutils.loads(response.text)
        self.assertEqual(metadata_tag_name, metadata_tag['name'])
        metadata_tag = jsonutils.loads(response.text)
        checked_keys = set(['name', 'created_at', 'updated_at'])
        self.assertEqual(checked_keys, set(metadata_tag.keys()))
        expected_metadata_tag = {'name': metadata_tag_name}
        checked_values = set(['name'])
        for key, value in expected_metadata_tag.items():
            if key in checked_values:
                self.assertEqual(metadata_tag[key], value, key)
        headers = self._headers({'content-type': 'application/json'})
        response = requests.post(path, headers=headers)
        self.assertEqual(http.CONFLICT, response.status_code)
        path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace_name, metadata_tag_name))
        media_type = 'application/json'
        headers = self._headers({'content-type': media_type})
        metadata_tag_name = 'tag1-UPDATED'
        data = jsonutils.dumps({'name': metadata_tag_name})
        response = requests.put(path, headers=headers, data=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        metadata_tag = jsonutils.loads(response.text)
        self.assertEqual('tag1-UPDATED', metadata_tag['name'])
        path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace_name, metadata_tag_name))
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertEqual('tag1-UPDATED', metadata_tag['name'])
        path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace_name, metadata_tag_name))
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.NO_CONTENT, response.status_code)
        path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace_name, metadata_tag_name))
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.NOT_FOUND, response.status_code)
        path = self._url('/v2/metadefs/namespaces/%s/tags' % namespace_name)
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'tags': [{'name': 'tag1'}, {'name': 'tag2'}, {'name': 'tag3'}]})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tags = jsonutils.loads(response.text)['tags']
        self.assertEqual(3, len(tags))
        data = jsonutils.dumps({'tags': [{'name': 'tag4'}, {'name': 'tag5'}, {'name': 'tag4'}]})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CONFLICT, response.status_code)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tags = jsonutils.loads(response.text)['tags']
        self.assertEqual(3, len(tags))
        path = self._url('/v2/metadefs/namespaces/%s/tags' % namespace_name)
        headers = self._headers({'content-type': 'application/json', 'X-Openstack-Append': 'True'})
        data = jsonutils.dumps({'tags': [{'name': 'tag4'}, {'name': 'tag5'}, {'name': 'tag6'}]})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tags = jsonutils.loads(response.text)['tags']
        self.assertEqual(6, len(tags))
        data = jsonutils.dumps({'tags': [{'name': 'tag6'}, {'name': 'tag7'}, {'name': 'tag8'}]})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CONFLICT, response.status_code)
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tags = jsonutils.loads(response.text)['tags']
        self.assertEqual(6, len(tags))

    def _create_tags(self, namespaces):
        tags = []
        for namespace in namespaces:
            headers = self._headers({'X-Tenant-Id': namespace['owner']})
            tag_name = 'tag_of_%s' % namespace['namespace']
            path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace['namespace'], tag_name))
            response = requests.post(path, headers=headers)
            self.assertEqual(http.CREATED, response.status_code)
            tag_metadata = response.json()
            metadef_tags = dict()
            metadef_tags[namespace['namespace']] = tag_metadata['name']
            tags.append(metadef_tags)
        return tags

    def _update_tags(self, path, headers, data):
        response = requests.put(path, headers=headers, json=data)
        self.assertEqual(http.OK, response.status_code, response.text)
        metadata_tag = response.json()
        self.assertEqual(data['name'], metadata_tag['name'])
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        self.assertEqual(data['name'], metadata_tag['name'])

    def test_role_base_metadata_tags_lifecycle(self):
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
        tenant1_tags = self._create_tags(tenant1_namespaces)
        tenant2_tags = self._create_tags(tenant2_namespaces)

        def _check_tag_access(tags, tenant):
            headers = self._headers({'content-type': 'application/json', 'X-Tenant-Id': tenant, 'X-Roles': 'reader,member'})
            for tag in tags:
                for namespace, tag_name in tag.items():
                    path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace, tag_name))
                    response = requests.get(path, headers=headers)
                    if namespace.split('_')[1] == 'public':
                        expected = http.OK
                    else:
                        expected = http.NOT_FOUND
                    self.assertEqual(expected, response.status_code)
                    path = self._url('/v2/metadefs/namespaces/%s/tags' % namespace)
                    response = requests.get(path, headers=headers)
                    self.assertEqual(expected, response.status_code)
                    if expected == http.OK:
                        resp_props = response.json()['tags']
                        self.assertEqual(sorted(tag.values()), sorted([x['name'] for x in resp_props]))
        _check_tag_access(tenant2_tags, self.tenant1)
        _check_tag_access(tenant1_tags, self.tenant2)
        total_tags = tenant1_tags + tenant2_tags
        for tag in total_tags:
            for namespace, tag_name in tag.items():
                data = {'name': tag_name}
                path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace, tag_name))
                headers['X-Roles'] = 'reader,member'
                response = requests.put(path, headers=headers, json=data)
                self.assertEqual(http.FORBIDDEN, response.status_code)
                headers = self._headers({'X-Tenant-Id': namespace.split('_')[0]})
                self._update_tags(path, headers, data)
        for tag in total_tags:
            for namespace, tag_name in tag.items():
                path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace, tag_name))
                response = requests.delete(path, headers=self._headers({'X-Roles': 'reader,member', 'X-Tenant-Id': namespace.split('_')[0]}))
                self.assertEqual(http.FORBIDDEN, response.status_code)
        headers = self._headers()
        for tag in total_tags:
            for namespace, tag_name in tag.items():
                path = self._url('/v2/metadefs/namespaces/%s/tags/%s' % (namespace, tag_name))
                response = requests.delete(path, headers=headers)
                self.assertEqual(http.NO_CONTENT, response.status_code)
                response = requests.get(path, headers=headers)
                self.assertEqual(http.NOT_FOUND, response.status_code)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
        data = {'tags': [{'name': 'tag1'}, {'name': 'tag2'}, {'name': 'tag3'}]}
        for namespace in tenant1_namespaces:
            path = self._url('/v2/metadefs/namespaces/%s/tags' % namespace['namespace'])
            response = requests.post(path, headers=headers, json=data)
            self.assertEqual(http.FORBIDDEN, response.status_code)
        headers = self._headers({'content-type': 'application/json'})
        for namespace in tenant1_namespaces:
            path = self._url('/v2/metadefs/namespaces/%s/tags' % namespace['namespace'])
            response = requests.post(path, headers=headers, json=data)
            self.assertEqual(http.CREATED, response.status_code)
        headers = self._headers({'content-type': 'application/json', 'X-Roles': 'reader,member'})
        for namespace in tenant1_namespaces:
            path = self._url('/v2/metadefs/namespaces/%s/tags' % namespace['namespace'])
            response = requests.delete(path, headers=headers)
            self.assertEqual(http.FORBIDDEN, response.status_code)
        headers = self._headers()
        for namespace in tenant1_namespaces:
            path = self._url('/v2/metadefs/namespaces/%s/tags' % namespace['namespace'])
            response = requests.delete(path, headers=headers)
            self.assertEqual(http.NO_CONTENT, response.status_code)