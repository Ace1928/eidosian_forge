from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
class WhenTestingACLEntity(ACLTestCase):

    def test_should_submit_acl_with_users_project_access_set(self, href=None):
        href = href or self.secret_ref
        data = {'acl_ref': self.secret_acl_ref}
        self.responses.put(self.secret_acl_ref, json=data)
        entity = self.manager.create(entity_ref=href + '///', users=self.users1, project_access=True)
        api_resp = entity.submit()
        self.assertEqual(self.secret_acl_ref, api_resp)
        self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)

    def test_should_submit_acl_with_users_project_access_stripped_uuid(self):
        bad_href = 'http://badsite.com/secrets/' + self.secret_uuid
        self.test_should_submit_acl_with_users_project_access_set(bad_href)

    def test_should_submit_acl_with_project_access_set_but_no_users(self):
        data = {'acl_ref': self.secret_acl_ref}
        self.responses.put(self.secret_acl_ref, json=data)
        entity = self.manager.create(entity_ref=self.secret_ref, project_access=False)
        api_resp = entity.submit()
        self.assertEqual(self.secret_acl_ref, api_resp)
        self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)
        self.assertFalse(entity.read.project_access)
        self.assertEqual([], entity.get('read').users)

    def test_should_submit_acl_with_user_set_but_not_project_access(self):
        data = {'acl_ref': self.container_acl_ref}
        self.responses.put(self.container_acl_ref, json=data)
        entity = self.manager.create(entity_ref=self.container_ref, users=self.users2)
        api_resp = entity.submit()
        self.assertEqual(self.container_acl_ref, api_resp)
        self.assertEqual(self.container_acl_ref, self.responses.last_request.url)
        self.assertEqual(self.users2, entity.read.users)
        self.assertIsNone(entity.get('read').project_access)

    def test_should_fail_submit_acl_invalid_secret_uri(self):
        data = {'acl_ref': self.secret_acl_ref}
        self.responses.put(self.secret_acl_ref, json=data)
        entity = self.manager.create(entity_ref=self.secret_acl_ref + '///', users=self.users1, project_access=True)
        self.assertRaises(ValueError, entity.submit)
        entity = self.manager.create(entity_ref=self.secret_ref, users=self.users1, project_access=True)
        entity._entity_ref = None
        self.assertRaises(ValueError, entity.submit)
        entity = self.manager.create(entity_ref=self.secret_ref, users=self.users1, project_access=True)
        entity._entity_ref = self.container_ref
        self.assertRaises(ValueError, entity.submit)

    def test_should_fail_submit_acl_invalid_container_uri(self):
        """Adding tests for container URI validation.

        Container URI validation is different from secret URI validation.
        That's why adding separate tests for code coverage.
        """
        data = {'acl_ref': self.container_acl_ref}
        self.responses.put(self.container_acl_ref, json=data)
        entity = self.manager.create(entity_ref=self.container_acl_ref + '///', users=self.users1, project_access=True)
        self.assertRaises(ValueError, entity.submit)
        entity = self.manager.create(entity_ref=self.container_ref, users=self.users1, project_access=True)
        entity._entity_ref = None
        self.assertRaises(ValueError, entity.submit)
        entity = self.manager.create(entity_ref=self.container_ref, users=self.users1, project_access=True)
        entity._entity_ref = self.secret_ref
        self.assertRaises(ValueError, entity.submit)

    def test_should_fail_submit_acl_no_acl_data(self):
        data = {'acl_ref': self.secret_acl_ref}
        self.responses.put(self.secret_acl_ref, json=data)
        entity = self.manager.create(entity_ref=self.secret_ref + '///')
        self.assertRaises(ValueError, entity.submit)

    def test_should_fail_submit_acl_input_users_as_not_list(self):
        data = {'acl_ref': self.secret_acl_ref}
        self.responses.put(self.secret_acl_ref, json=data)
        entity = self.manager.create(entity_ref=self.secret_ref, users='u1')
        self.assertRaises(ValueError, entity.submit)

    def test_should_load_acls_data(self):
        self.responses.get(self.container_acl_ref, json=self.get_acl_response_data(users=self.users2, project_access=True))
        entity = self.manager.create(entity_ref=self.container_ref, users=self.users1)
        self.assertEqual(self.container_ref, entity.entity_ref)
        self.assertEqual(self.container_acl_ref, entity.acl_ref)
        entity.load_acls_data()
        self.assertEqual(self.users2, entity.read.users)
        self.assertTrue(entity.get('read').project_access)
        self.assertEqual(timeutils.parse_isotime(self.created), entity.read.created)
        self.assertEqual(timeutils.parse_isotime(self.created), entity.get('read').created)
        self.assertEqual(1, len(entity.operation_acls))
        self.assertEqual(self.container_acl_ref, entity.get('read').acl_ref)
        self.assertEqual(self.container_ref, entity.read.entity_ref)

    def test_should_add_operation_acl(self):
        entity = self.manager.create(entity_ref=self.secret_ref + '///', users=self.users1, project_access=True)
        self.assertIsInstance(entity, acls.SecretACL)
        entity.add_operation_acl(users=self.users2, project_access=False, operation_type='read')
        read_acl = entity.read
        self.assertEqual(self.secret_ref + '/acl', read_acl.acl_ref)
        self.assertFalse(read_acl.project_access)
        self.assertEqual(self.users2, read_acl.users)
        self.assertEqual(acls.DEFAULT_OPERATION_TYPE, read_acl.operation_type)
        entity.add_operation_acl(users=[], project_access=False, operation_type='dummy')
        dummy_acl = entity.get('dummy')
        self.assertFalse(dummy_acl.project_access)
        self.assertEqual([], dummy_acl.users)

    def test_acl_entity_properties(self):
        entity = self.manager.create(entity_ref=self.secret_ref, users=self.users2)
        self.assertEqual(self.secret_ref, entity.entity_ref)
        self.assertEqual(self.secret_acl_ref, entity.acl_ref)
        self.assertEqual(self.users2, entity.read.users)
        self.assertEqual(self.users2, entity.get('read').users)
        self.assertIsNone(entity.read.project_access)
        self.assertIsNone(entity.get('read').project_access)
        self.assertIsNone(entity.read.created)
        self.assertIsNone(entity.get('read').created)
        self.assertEqual('read', entity.read.operation_type)
        self.assertEqual('read', entity.get('read').operation_type)
        self.assertEqual(1, len(entity.operation_acls))
        self.assertEqual(self.secret_acl_ref, entity.read.acl_ref)
        self.assertEqual(self.secret_acl_ref, entity.get('read').acl_ref)
        self.assertEqual(self.secret_ref, entity.read.entity_ref)
        self.assertIsNone(entity.get('dummyOperation'))
        entity.read.users = ['u1']
        entity.read.project_access = False
        entity.read.operation_type = 'my_operation'
        self.assertFalse(entity.get('my_operation').project_access)
        self.assertEqual(['u1'], entity.get('my_operation').users)
        self.assertRaises(AttributeError, lambda x: x.dummy_operation, entity)

    def test_get_formatted_data(self):
        s_entity = acls.SecretACL(api=None, entity_ref=self.secret_ref, users=self.users1)
        data = s_entity.read._get_formatted_data()
        self.assertEqual(acls.DEFAULT_OPERATION_TYPE, data[0])
        self.assertIsNone(data[1])
        self.assertEqual(self.users1, data[2])
        self.assertIsNone(data[3])
        self.assertIsNone(data[4])
        self.assertEqual(self.secret_acl_ref, data[5])
        c_entity = acls.ContainerACL(api=None, entity_ref=self.container_ref, users=self.users2, created=self.created)
        data = c_entity.get('read')._get_formatted_data()
        self.assertEqual(acls.DEFAULT_OPERATION_TYPE, data[0])
        self.assertIsNone(data[1])
        self.assertEqual(self.users2, data[2])
        self.assertEqual(timeutils.parse_isotime(self.created).isoformat(), data[3])
        self.assertIsNone(data[4])
        self.assertEqual(self.container_acl_ref, data[5])

    def test_should_secret_acl_remove(self, entity_ref=None):
        entity_ref = entity_ref or self.secret_ref
        self.responses.delete(self.secret_acl_ref)
        entity = self.manager.create(entity_ref=entity_ref, users=self.users2)
        api_resp = entity.remove()
        self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)
        self.assertIsNone(api_resp)

    def test_should_secret_acl_remove_uri_with_slashes(self):
        self.responses.delete(self.secret_acl_ref)
        entity = self.manager.create(entity_ref=self.secret_ref + '///', users=self.users2)
        entity.remove()
        self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)
        self.responses.delete(self.container_acl_ref)

    def test_should_secret_acl_remove_stripped_uuid(self):
        bad_href = 'http://badsite.com/secrets/' + self.secret_uuid
        self.test_should_secret_acl_remove(bad_href)

    def test_should_container_acl_remove(self, entity_ref=None):
        entity_ref = entity_ref or self.container_ref
        self.responses.delete(self.container_acl_ref)
        entity = self.manager.create(entity_ref=entity_ref)
        entity.remove()
        self.assertEqual(self.container_acl_ref, self.responses.last_request.url)

    def test_should_container_acl_remove_stripped_uuid(self):
        bad_href = 'http://badsite.com/containers/' + self.container_uuid
        self.test_should_container_acl_remove(bad_href)

    def test_should_fail_acl_remove_invalid_uri(self):
        entity = self.manager.create(entity_ref=self.secret_acl_ref)
        self.assertRaises(ValueError, entity.remove)
        entity = self.manager.create(entity_ref=self.container_acl_ref)
        self.assertRaises(ValueError, entity.remove)
        entity = self.manager.create(entity_ref=self.container_ref + '/consumers')
        self.assertRaises(ValueError, entity.remove)
        entity = self.manager.create(entity_ref=self.endpoint + '/secrets' + '/consumers')
        self.assertRaises(ValueError, entity.remove)

    def test_should_per_operation_acl_remove(self):
        self.responses.get(self.secret_acl_ref, json=self.get_acl_response_data(users=self.users2, project_access=True))
        self.responses.delete(self.secret_acl_ref)
        entity = self.manager.create(entity_ref=self.secret_ref, users=self.users2)
        api_resp = entity.read.remove()
        self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)
        self.assertIsNone(api_resp)
        self.assertEqual(0, len(entity.operation_acls))
        acl_data = self.get_acl_response_data(users=self.users2, project_access=True)
        data = self.get_acl_response_data(users=self.users1, operation_type='write', project_access=False)
        acl_data['write'] = data['write']
        self.responses.get(self.secret_acl_ref, json=acl_data)
        self.responses.put(self.secret_acl_ref, json={})
        entity = self.manager.create(entity_ref=self.secret_ref, users=self.users2)
        entity.read.remove()
        self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)
        self.assertEqual(1, len(entity.operation_acls))
        self.assertEqual('write', entity.operation_acls[0].operation_type)
        self.assertEqual(self.users1, entity.operation_acls[0].users)