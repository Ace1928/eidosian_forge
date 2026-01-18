from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
class TestMetadefTagsPolicy(functional.SynchronousAPIBase):

    def setUp(self):
        super(TestMetadefTagsPolicy, self).setUp()
        self.policy = policy.Enforcer(suppress_deprecation_warnings=True)

    def load_data(self, create_tags=False):
        path = '/v2/metadefs/namespaces'
        md_resource = self._create_metadef_resource(path=path, data=NAME_SPACE1)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        if create_tags:
            namespace = md_resource['namespace']
            for tag in [TAG1, TAG2]:
                path = '/v2/metadefs/namespaces/%s/tags/%s' % (namespace, tag['name'])
                md_resource = self._create_metadef_resource(path=path)
                self.assertEqual(tag['name'], md_resource['name'])

    def set_policy_rules(self, rules):
        self.policy.set_rules(oslo_policy.policy.Rules.from_dict(rules), overwrite=True)

    def start_server(self):
        with mock.patch.object(policy, 'Enforcer') as mock_enf:
            mock_enf.return_value = self.policy
            super(TestMetadefTagsPolicy, self).start_server()

    def _verify_forbidden_converted_to_not_found(self, path, method, json=None):
        headers = self._headers({'X-Tenant-Id': 'fake-tenant-id', 'X-Roles': 'member'})
        resp = self.api_request(method, path, headers=headers, json=json)
        self.assertEqual(404, resp.status_code)

    def test_tag_create_basic(self):
        self.start_server()
        self.load_data()
        namespace = NAME_SPACE1['namespace']
        path = '/v2/metadefs/namespaces/%s/tags/%s' % (namespace, TAG1['name'])
        md_resource = self._create_metadef_resource(path=path)
        self.assertEqual('MyTag', md_resource['name'])
        self.set_policy_rules({'add_metadef_tag': '!', 'get_metadef_namespace': ''})
        path = '/v2/metadefs/namespaces/%s/tags/%s' % (namespace, TAG2['name'])
        resp = self.api_post(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'add_metadef_tag': '!', 'get_metadef_namespace': '!'})
        resp = self.api_post(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'add_metadef_tag': '', 'get_metadef_namespace': ''})
        self._verify_forbidden_converted_to_not_found(path, 'POST')

    def test_tags_create_basic(self):
        self.start_server()
        self.load_data()
        namespace = NAME_SPACE1['namespace']
        path = '/v2/metadefs/namespaces/%s/tags' % namespace
        data = {'tags': [TAG1, TAG2]}
        md_resource = self._create_metadef_resource(path=path, data=data)
        self.assertEqual(2, len(md_resource['tags']))
        self.set_policy_rules({'add_metadef_tags': '!', 'get_metadef_namespace': ''})
        path = '/v2/metadefs/namespaces/%s/tags' % namespace
        data = {'tags': [{'name': 'sampe-tag-1'}, {'name': 'sampe-tag-2'}]}
        resp = self.api_post(path, json=data)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'add_metadef_tags': '!', 'get_metadef_namespace': '!'})
        resp = self.api_post(path, json=data)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'add_metadef_tags': '', 'get_metadef_namespace': ''})
        self._verify_forbidden_converted_to_not_found(path, 'POST', json=data)

    def test_tag_list_basic(self):
        self.start_server()
        self.load_data(create_tags=True)
        namespace = NAME_SPACE1['namespace']
        path = '/v2/metadefs/namespaces/%s/tags' % namespace
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual(2, len(md_resource['tags']))
        self.set_policy_rules({'get_metadef_tags': '!', 'get_metadef_namespace': ''})
        resp = self.api_get(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_metadef_tags': '!', 'get_metadef_namespace': '!'})
        resp = self.api_get(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'get_metadef_tags': '', 'get_metadef_namespace': ''})
        self._verify_forbidden_converted_to_not_found(path, 'GET')

    def test_tag_get_basic(self):
        self.start_server()
        self.load_data(create_tags=True)
        namespace = NAME_SPACE1['namespace']
        path = '/v2/metadefs/namespaces/%s/tags/%s' % (namespace, TAG1['name'])
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual('MyTag', md_resource['name'])
        self.set_policy_rules({'get_metadef_tag': '!', 'get_metadef_namespace': ''})
        resp = self.api_get(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_metadef_tag': '!', 'get_metadef_namespace': '!'})
        resp = self.api_get(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'get_metadef_tag': '', 'get_metadef_namespace': ''})
        self._verify_forbidden_converted_to_not_found(path, 'GET')

    def test_tag_update_basic(self):
        self.start_server()
        self.load_data(create_tags=True)
        namespace = NAME_SPACE1['namespace']
        path = '/v2/metadefs/namespaces/%s/tags/%s' % (namespace, TAG1['name'])
        data = {'name': 'MyTagUpdated'}
        resp = self.api_put(path, json=data)
        md_resource = resp.json
        self.assertEqual('MyTagUpdated', md_resource['name'])
        self.set_policy_rules({'modify_metadef_tag': '!', 'get_metadef_namespace': ''})
        path = '/v2/metadefs/namespaces/%s/tags/%s' % (namespace, TAG2['name'])
        data = {'name': 'MySecondTagUpdated'}
        resp = self.api_put(path, json=data)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'modify_metadef_tag': '!', 'get_metadef_namespace': '!'})
        resp = self.api_put(path, json=data)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'get_metadef_tag': '', 'get_metadef_namespace': ''})
        self._verify_forbidden_converted_to_not_found(path, 'PUT', json=data)

    def test_tag_delete_basic(self):
        self.start_server()
        self.load_data(create_tags=True)
        path = '/v2/metadefs/namespaces/%s/tags/%s' % (NAME_SPACE1['namespace'], TAG1['name'])
        resp = self.api_delete(path)
        self.assertEqual(204, resp.status_code)
        path = '/v2/metadefs/namespaces/%s/tags/%s' % (NAME_SPACE1['namespace'], TAG1['name'])
        resp = self.api_get(path)
        self.assertEqual(404, resp.status_code)
        path = '/v2/metadefs/namespaces/%s/tags/%s' % (NAME_SPACE1['namespace'], TAG2['name'])
        self.set_policy_rules({'delete_metadef_tag': '!', 'get_metadef_namespace': ''})
        resp = self.api_delete(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'delete_metadef_tag': '!', 'get_metadef_namespace': '!'})
        resp = self.api_delete(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'delete_metadef_tag': '', 'get_metadef_namespace': ''})
        self._verify_forbidden_converted_to_not_found(path, 'DELETE')

    def test_tags_delete_basic(self):
        self.start_server()
        self.load_data(create_tags=True)
        path = '/v2/metadefs/namespaces/%s/tags' % NAME_SPACE1['namespace']
        resp = self.api_delete(path)
        self.assertEqual(204, resp.status_code)
        path = '/v2/metadefs/namespaces/%s/tags' % NAME_SPACE1['namespace']
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual(0, len(md_resource['tags']))
        path = '/v2/metadefs/namespaces/%s/tags' % NAME_SPACE1['namespace']
        self.set_policy_rules({'delete_metadef_tags': '!', 'get_metadef_namespace': ''})
        resp = self.api_delete(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'delete_metadef_tags': '!', 'get_metadef_namespace': '!'})
        resp = self.api_delete(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'delete_metadef_tags': '', 'get_metadef_namespace': ''})
        self._verify_forbidden_converted_to_not_found(path, 'DELETE')