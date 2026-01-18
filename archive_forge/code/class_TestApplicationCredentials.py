from openstack import exceptions
from openstack.tests.functional import base
class TestApplicationCredentials(base.BaseFunctionalTest):

    def setUp(self):
        super(TestApplicationCredentials, self).setUp()
        self.user_id = self.operator_cloud.current_user_id

    def _create_application_credentials(self):
        app_creds = self.conn.identity.create_application_credential(user=self.user_id, name='app_cred')
        self.addCleanup(self.conn.identity.delete_application_credential, self.user_id, app_creds['id'])
        return app_creds

    def test_create_application_credentials(self):
        app_creds = self._create_application_credentials()
        self.assertEqual(app_creds['user_id'], self.user_id)

    def test_get_application_credential(self):
        app_creds = self._create_application_credentials()
        app_cred = self.conn.identity.get_application_credential(user=self.user_id, application_credential=app_creds['id'])
        self.assertEqual(app_cred['id'], app_creds['id'])
        self.assertEqual(app_cred['user_id'], self.user_id)

    def test_application_credentials(self):
        self._create_application_credentials()
        app_creds = self.conn.identity.application_credentials(user=self.user_id)
        for app_cred in app_creds:
            self.assertEqual(app_cred['user_id'], self.user_id)

    def test_find_application_credential(self):
        app_creds = self._create_application_credentials()
        app_cred = self.conn.identity.find_application_credential(user=self.user_id, name_or_id=app_creds['id'])
        self.assertEqual(app_cred['id'], app_creds['id'])
        self.assertEqual(app_cred['user_id'], self.user_id)

    def test_delete_application_credential(self):
        app_creds = self._create_application_credentials()
        self.conn.identity.delete_application_credential(user=self.user_id, application_credential=app_creds['id'])
        self.assertRaises(exceptions.NotFoundException, self.conn.identity.get_application_credential, user=self.user_id, application_credential=app_creds['id'])