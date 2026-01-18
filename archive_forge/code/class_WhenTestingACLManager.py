from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
class WhenTestingACLManager(ACLTestCase):

    def test_should_get_secret_acl(self, entity_ref=None):
        entity_ref = entity_ref or self.secret_ref
        self.responses.get(self.secret_acl_ref, json=self.get_acl_response_data())
        api_resp = self.manager.get(entity_ref=entity_ref)
        self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)
        self.assertFalse(api_resp.get('read').project_access)
        self.assertEqual('read', api_resp.get('read').operation_type)
        self.assertIn(api_resp.get('read').acl_ref_relative, self.secret_acl_ref)

    def test_should_get_secret_acl_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/secrets/' + self.secret_uuid
        self.test_should_get_secret_acl(bad_href)

    def test_should_get_secret_acl_with_extra_trailing_slashes(self):
        self.responses.get(requests_mock.ANY, json=self.get_acl_response_data())
        self.manager.get(entity_ref=self.secret_ref + '///')
        self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)

    def test_should_get_container_acl(self, entity_ref=None):
        entity_ref = entity_ref or self.container_ref
        self.responses.get(self.container_acl_ref, json=self.get_acl_response_data())
        api_resp = self.manager.get(entity_ref=entity_ref)
        self.assertEqual(self.container_acl_ref, self.responses.last_request.url)
        self.assertFalse(api_resp.get('read').project_access)
        self.assertEqual('read', api_resp.get('read').operation_type)
        self.assertIn(api_resp.get('read').acl_ref_relative, self.container_acl_ref)

    def test_should_get_container_acl_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/containers/' + self.container_uuid
        self.test_should_get_container_acl(bad_href)

    def test_should_get_container_acl_with_trailing_slashes(self):
        self.responses.get(requests_mock.ANY, json=self.get_acl_response_data())
        self.manager.get(entity_ref=self.container_ref + '///')
        self.assertEqual(self.container_acl_ref, self.responses.last_request.url)

    def test_should_fail_get_no_href(self):
        self.assertRaises(ValueError, self.manager.get, None)

    def test_should_fail_get_invalid_uri(self):
        self.assertRaises(ValueError, self.manager.get, self.secret_acl_ref)
        self.assertRaises(ValueError, self.manager.get, self.endpoint + '/containers/consumers')

    def test_should_create_secret_acl(self):
        entity = self.manager.create(entity_ref=self.secret_ref + '///', users=self.users1, project_access=True)
        self.assertIsInstance(entity, acls.SecretACL)
        read_acl = entity.read
        self.assertEqual(self.secret_ref + '///', read_acl.entity_ref)
        self.assertTrue(read_acl.project_access)
        self.assertEqual(self.users1, read_acl.users)
        self.assertEqual(acls.DEFAULT_OPERATION_TYPE, read_acl.operation_type)
        self.assertIn(self.secret_ref, read_acl.acl_ref, 'ACL ref has additional /acl')
        self.assertIsNone(read_acl.created)
        self.assertIsNone(read_acl.updated)
        read_acl_via_get = entity.get('read')
        self.assertEqual(read_acl, read_acl_via_get)

    def test_should_create_acl_with_users(self, entity_ref=None):
        entity_ref = entity_ref or self.container_ref
        entity = self.manager.create(entity_ref=entity_ref + '///', users=self.users2, project_access=False)
        self.assertIsInstance(entity, acls.ContainerACL)
        self.assertEqual(entity_ref + '///', entity.entity_ref)
        read_acl = entity.read
        self.assertFalse(read_acl.project_access)
        self.assertEqual(self.users2, read_acl.users)
        self.assertEqual(acls.DEFAULT_OPERATION_TYPE, read_acl.operation_type)
        self.assertIn(entity_ref, read_acl.acl_ref, 'ACL ref has additional /acl')
        self.assertIn(read_acl.acl_ref_relative, self.container_acl_ref)

    def test_should_create_acl_with_users_stripped_uuid(self):
        bad_href = 'http://badsite.com/containers/' + self.container_uuid
        self.test_should_create_acl_with_users(bad_href)

    def test_should_create_acl_with_no_users(self):
        entity = self.manager.create(entity_ref=self.container_ref, users=[])
        read_acl = entity.read
        self.assertEqual([], read_acl.users)
        self.assertEqual(acls.DEFAULT_OPERATION_TYPE, read_acl.operation_type)
        self.assertIsNone(read_acl.project_access)
        read_acl_via_get = entity.get('read')
        self.assertEqual(read_acl, read_acl_via_get)

    def test_create_no_acl_settings(self):
        entity = self.manager.create(entity_ref=self.container_ref)
        self.assertEqual([], entity.operation_acls)
        self.assertEqual(self.container_ref, entity.entity_ref)
        self.assertEqual(self.container_ref + '/acl', entity.acl_ref)

    def test_should_fail_create_invalid_uri(self):
        self.assertRaises(ValueError, self.manager.create, self.endpoint + '/orders')
        self.assertRaises(ValueError, self.manager.create, None)