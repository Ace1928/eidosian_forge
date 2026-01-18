import base64
import subprocess
from unittest import mock
from novaclient import crypto
from novaclient.tests.unit import utils
class CryptoTest(utils.TestCase):

    def setUp(self):
        super(CryptoTest, self).setUp()
        self.password_string = 'Test Password'
        self.decrypt_password = b'Decrypt Password'
        self.private_key = 'Test Private Key'

    @mock.patch('subprocess.Popen')
    def test_decrypt_password(self, mock_open):
        mocked_proc = mock.Mock()
        mock_open.return_value = mocked_proc
        mocked_proc.returncode = 0
        mocked_proc.communicate.return_value = (self.decrypt_password, '')
        decrypt_password = crypto.decrypt_password(self.private_key, self.password_string)
        self.assertIsInstance(decrypt_password, str)
        self.assertEqual('Decrypt Password', decrypt_password)
        mock_open.assert_called_once_with(['openssl', 'rsautl', '-decrypt', '-inkey', self.private_key], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        mocked_proc.communicate.assert_called_once_with(base64.b64decode(self.password_string))
        mocked_proc.stdin.close.assert_called_once_with()

    @mock.patch('subprocess.Popen')
    def test_decrypt_password_failure(self, mock_open):
        mocked_proc = mock.Mock()
        mock_open.return_value = mocked_proc
        mocked_proc.returncode = 1
        mocked_proc.communicate.return_value = (self.decrypt_password, '')
        self.assertRaises(crypto.DecryptionFailure, crypto.decrypt_password, self.private_key, self.password_string)
        mock_open.assert_called_once_with(['openssl', 'rsautl', '-decrypt', '-inkey', self.private_key], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        mocked_proc.communicate.assert_called_once_with(base64.b64decode(self.password_string))
        mocked_proc.stdin.close.assert_called_once_with()