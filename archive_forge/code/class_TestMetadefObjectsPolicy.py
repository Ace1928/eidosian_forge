from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
class TestMetadefObjectsPolicy(functional.SynchronousAPIBase):

    def setUp(self):
        super(TestMetadefObjectsPolicy, self).setUp()
        self.policy = policy.Enforcer(suppress_deprecation_warnings=True)

    def load_data(self, create_objects=False):
        path = '/v2/metadefs/namespaces'
        md_resource = self._create_metadef_resource(path=path, data=NAME_SPACE1)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        if create_objects:
            namespace = md_resource['namespace']
            path = '/v2/metadefs/namespaces/%s/objects' % namespace
            for obj in [OBJECT1, OBJECT2]:
                md_resource = self._create_metadef_resource(path=path, data=obj)
                self.assertEqual(obj['name'], md_resource['name'])

    def set_policy_rules(self, rules):
        self.policy.set_rules(oslo_policy.policy.Rules.from_dict(rules), overwrite=True)

    def start_server(self):
        with mock.patch.object(policy, 'Enforcer') as mock_enf:
            mock_enf.return_value = self.policy
            super(TestMetadefObjectsPolicy, self).start_server()

    def _verify_forbidden_converted_to_not_found(self, path, method, json=None):
        headers = self._headers({'X-Tenant-Id': 'fake-tenant-id', 'X-Roles': 'member'})
        resp = self.api_request(method, path, headers=headers, json=json)
        self.assertEqual(404, resp.status_code)

    def test_object_create_basic(self):
        self.start_server()
        self.load_data()
        path = '/v2/metadefs/namespaces/%s/objects' % NAME_SPACE1['namespace']
        md_resource = self._create_metadef_resource(path=path, data=OBJECT1)
        self.assertEqual('MyObject', md_resource['name'])
        self.set_policy_rules({'add_metadef_object': '!', 'get_metadef_namespace': '@'})
        resp = self.api_post(path, json=OBJECT2)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'add_metadef_object': '!', 'get_metadef_namespace': '!'})
        resp = self.api_post(path, json=OBJECT2)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'add_metadef_object': '@', 'get_metadef_namespace': '@'})
        self._verify_forbidden_converted_to_not_found(path, 'POST', json=OBJECT2)

    def test_object_list_basic(self):
        self.start_server()
        self.load_data(create_objects=True)
        path = '/v2/metadefs/namespaces/%s/objects' % NAME_SPACE1['namespace']
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual(2, len(md_resource['objects']))
        self.set_policy_rules({'get_metadef_objects': '!', 'get_metadef_namespace': '@'})
        resp = self.api_get(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_metadef_objects': '!', 'get_metadef_namespace': '!'})
        resp = self.api_get(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'get_metadef_objects': '@', 'get_metadef_object': '!', 'get_metadef_namespace': '@'})
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual(0, len(md_resource['objects']))
        self.set_policy_rules({'get_metadef_objects': '@', 'get_metadef_object': '@', 'get_metadef_namespace': '@'})
        self._verify_forbidden_converted_to_not_found(path, 'GET')

    def test_object_get_basic(self):
        self.start_server()
        self.load_data(create_objects=True)
        path = '/v2/metadefs/namespaces/%s/objects/%s' % (NAME_SPACE1['namespace'], OBJECT1['name'])
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual(OBJECT1['name'], md_resource['name'])
        self.set_policy_rules({'get_metadef_object': '!', 'get_metadef_namespace': '@'})
        resp = self.api_get(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_metadef_object': '!', 'get_metadef_namespace': '!'})
        resp = self.api_get(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'get_metadef_object': '@', 'get_metadef_namespace': '@'})
        self._verify_forbidden_converted_to_not_found(path, 'GET')

    def test_object_update_basic(self):
        self.start_server()
        self.load_data(create_objects=True)
        path = '/v2/metadefs/namespaces/%s/objects/%s' % (NAME_SPACE1['namespace'], OBJECT1['name'])
        data = {'name': OBJECT1['name'], 'description': 'My updated description'}
        resp = self.api_put(path, json=data)
        md_resource = resp.json
        self.assertEqual(data['description'], md_resource['description'])
        data = {'name': OBJECT2['name'], 'description': 'My updated description'}
        path = '/v2/metadefs/namespaces/%s/objects/%s' % (NAME_SPACE1['namespace'], OBJECT2['name'])
        self.set_policy_rules({'modify_metadef_object': '!', 'get_metadef_namespace': '@'})
        resp = self.api_put(path, json=data)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'modify_metadef_object': '!', 'get_metadef_namespace': '!'})
        resp = self.api_put(path, json=data)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'modify_metadef_object': '@', 'get_metadef_namespace': '@'})
        self._verify_forbidden_converted_to_not_found(path, 'PUT', json=data)

    def test_object_delete_basic(self):
        self.start_server()
        self.load_data(create_objects=True)
        path = '/v2/metadefs/namespaces/%s/objects/%s' % (NAME_SPACE1['namespace'], OBJECT1['name'])
        resp = self.api_delete(path)
        self.assertEqual(204, resp.status_code)
        path = '/v2/metadefs/namespaces/%s/objects/%s' % (NAME_SPACE1['namespace'], OBJECT1['name'])
        resp = self.api_get(path)
        self.assertEqual(404, resp.status_code)
        path = '/v2/metadefs/namespaces/%s/objects/%s' % (NAME_SPACE1['namespace'], OBJECT2['name'])
        self.set_policy_rules({'delete_metadef_object': '!', 'get_metadef_namespace': '@'})
        resp = self.api_delete(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'delete_metadef_object': '!', 'get_metadef_namespace': '!'})
        resp = self.api_delete(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'delete_metadef_object': '@', 'get_metadef_namespace': '@'})
        self._verify_forbidden_converted_to_not_found(path, 'DELETE')