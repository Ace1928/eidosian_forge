from castellan.common.credentials import password
from castellan.tests import base
class PasswordTestCase(base.TestCase):

    def _create_password_credential(self):
        return password.Password(self.username, self.password)

    def setUp(self):
        self.username = 'admin'
        self.password = 'Pa$$w0rd1'
        self.password_credential = self._create_password_credential()
        super(PasswordTestCase, self).setUp()

    def test_get_username(self):
        self.assertEqual(self.username, self.password_credential.username)

    def test_get_password(self):
        self.assertEqual(self.password, self.password_credential.password)

    def test___eq__(self):
        self.assertTrue(self.password_credential == self.password_credential)
        self.assertTrue(self.password_credential is self.password_credential)
        self.assertFalse(self.password_credential is None)
        self.assertFalse(None == self.password_credential)
        other_password_credential = password.Password(self.username, self.password)
        self.assertTrue(self.password_credential == other_password_credential)
        self.assertFalse(self.password_credential is other_password_credential)

    def test___ne___none(self):
        self.assertTrue(self.password_credential is not None)
        self.assertTrue(None != self.password_credential)

    def test___ne___username(self):
        other_username = 'service'
        other_password_credential = password.Password(other_username, self.password)
        self.assertTrue(self.password_credential != other_password_credential)

    def test___ne___password(self):
        other_password = 'i143Cats'
        other_password_credential = password.Password(self.username, other_password)
        self.assertTrue(self.password_credential != other_password_credential)