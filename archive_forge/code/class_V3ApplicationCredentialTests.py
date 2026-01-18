import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
class V3ApplicationCredentialTests(utils.TestCase):

    def setUp(self):
        super(V3ApplicationCredentialTests, self).setUp()
        self.auth_url = uuid.uuid4().hex

    def create(self, **kwargs):
        kwargs.setdefault('auth_url', self.auth_url)
        loader = loading.get_plugin_loader('v3applicationcredential')
        return loader.load_from_options(**kwargs)

    def test_basic(self):
        id = uuid.uuid4().hex
        secret = uuid.uuid4().hex
        app_cred = self.create(application_credential_id=id, application_credential_secret=secret)
        ac_method = app_cred.auth_methods[0]
        self.assertEqual(id, ac_method.application_credential_id)
        self.assertEqual(secret, ac_method.application_credential_secret)

    def test_with_name(self):
        name = uuid.uuid4().hex
        secret = uuid.uuid4().hex
        username = uuid.uuid4().hex
        user_domain_id = uuid.uuid4().hex
        app_cred = self.create(application_credential_name=name, application_credential_secret=secret, username=username, user_domain_id=user_domain_id)
        ac_method = app_cred.auth_methods[0]
        self.assertEqual(name, ac_method.application_credential_name)
        self.assertEqual(secret, ac_method.application_credential_secret)
        self.assertEqual(username, ac_method.username)
        self.assertEqual(user_domain_id, ac_method.user_domain_id)

    def test_without_user_domain(self):
        self.assertRaises(exceptions.OptionError, self.create, application_credential_name=uuid.uuid4().hex, username=uuid.uuid4().hex, application_credential_secret=uuid.uuid4().hex)

    def test_without_name_or_id(self):
        self.assertRaises(exceptions.OptionError, self.create, username=uuid.uuid4().hex, user_domain_id=uuid.uuid4().hex, application_credential_secret=uuid.uuid4().hex)

    def test_without_secret(self):
        self.assertRaises(exceptions.OptionError, self.create, application_credential_id=uuid.uuid4().hex, username=uuid.uuid4().hex, user_domain_id=uuid.uuid4().hex)