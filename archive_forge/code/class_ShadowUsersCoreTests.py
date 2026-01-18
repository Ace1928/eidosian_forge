import copy
import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
class ShadowUsersCoreTests(object):

    def test_shadow_federated_user(self):
        federated_user1 = copy.deepcopy(self.federated_user)
        ShadowUsersCoreTests.normalize_federated_user_properties_for_test(federated_user1, email=self.email)
        user = PROVIDERS.identity_api.shadow_federated_user(self.federated_user['idp_id'], self.federated_user['protocol_id'], federated_user1)
        self.assertIsNotNone(user['id'])
        self.assertEqual(7, len(user.keys()))
        self.assertIsNotNone(user['name'])
        self.assertIsNone(user['password_expires_at'])
        self.assertIsNotNone(user['domain_id'])
        self.assertEqual(True, user['enabled'])
        self.assertIsNotNone(user['email'])

    def test_shadow_existing_federated_user(self):
        federated_user1 = copy.deepcopy(self.federated_user)
        ShadowUsersCoreTests.normalize_federated_user_properties_for_test(federated_user1, email=self.email)
        shadow_user1 = PROVIDERS.identity_api.shadow_federated_user(self.federated_user['idp_id'], self.federated_user['protocol_id'], federated_user1)
        self.assertEqual(federated_user1['display_name'], shadow_user1['name'])
        federated_user2 = copy.deepcopy(self.federated_user)
        federated_user2['display_name'] = uuid.uuid4().hex
        ShadowUsersCoreTests.normalize_federated_user_properties_for_test(federated_user2, email=self.email)
        shadow_user2 = PROVIDERS.identity_api.shadow_federated_user(self.federated_user['idp_id'], self.federated_user['protocol_id'], federated_user2)
        self.assertEqual(federated_user2['display_name'], shadow_user2['name'])
        self.assertNotEqual(shadow_user1['name'], shadow_user2['name'])
        self.assertEqual(shadow_user1['id'], shadow_user2['id'])

    def test_shadow_federated_user_not_creating_a_local_user(self):
        federated_user1 = copy.deepcopy(self.federated_user)
        ShadowUsersCoreTests.normalize_federated_user_properties_for_test(federated_user1, email='some_id@mail.provider')
        PROVIDERS.identity_api.shadow_federated_user(federated_user1['idp_id'], federated_user1['protocol_id'], federated_user1)
        hints = driver_hints.Hints()
        hints.add_filter('name', federated_user1['display_name'])
        users = PROVIDERS.identity_api.list_users(hints=hints)
        self.assertEqual(1, len(users))
        federated_user2 = copy.deepcopy(federated_user1)
        federated_user2['name'] = uuid.uuid4().hex
        federated_user2['id'] = uuid.uuid4().hex
        federated_user2['email'] = 'some_id_2@mail.provider'
        PROVIDERS.identity_api.shadow_federated_user(federated_user2['idp_id'], federated_user2['protocol_id'], federated_user2)
        hints.add_filter('name', federated_user2['display_name'])
        users = PROVIDERS.identity_api.list_users(hints=hints)
        self.assertEqual(1, len(users))

    @staticmethod
    def normalize_federated_user_properties_for_test(federated_user, email=None):
        federated_user['email'] = email
        federated_user['id'] = federated_user['unique_id']
        federated_user['name'] = federated_user['display_name']
        if not federated_user.get('domain'):
            federated_user['domain'] = {'id': uuid.uuid4().hex}