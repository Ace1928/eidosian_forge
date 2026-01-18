import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
class TestRunCertProviderCommand(object):

    def create_mock_process(self, output, error):
        mock_process = mock.Mock()
        attrs = {'communicate.return_value': (output, error), 'returncode': 0}
        mock_process.configure_mock(**attrs)
        return mock_process

    @mock.patch('subprocess.Popen', autospec=True)
    def test_success(self, mock_popen):
        mock_popen.return_value = self.create_mock_process(pytest.public_cert_bytes + pytest.private_key_bytes, b'')
        cert, key, passphrase = _mtls_helper._run_cert_provider_command(['command'])
        assert cert == pytest.public_cert_bytes
        assert key == pytest.private_key_bytes
        assert passphrase is None
        mock_popen.return_value = self.create_mock_process(pytest.public_cert_bytes + ENCRYPTED_EC_PRIVATE_KEY + PASSPHRASE, b'')
        cert, key, passphrase = _mtls_helper._run_cert_provider_command(['command'], expect_encrypted_key=True)
        assert cert == pytest.public_cert_bytes
        assert key == ENCRYPTED_EC_PRIVATE_KEY
        assert passphrase == PASSPHRASE_VALUE

    @mock.patch('subprocess.Popen', autospec=True)
    def test_success_with_cert_chain(self, mock_popen):
        PUBLIC_CERT_CHAIN_BYTES = pytest.public_cert_bytes + pytest.public_cert_bytes
        mock_popen.return_value = self.create_mock_process(PUBLIC_CERT_CHAIN_BYTES + pytest.private_key_bytes, b'')
        cert, key, passphrase = _mtls_helper._run_cert_provider_command(['command'])
        assert cert == PUBLIC_CERT_CHAIN_BYTES
        assert key == pytest.private_key_bytes
        assert passphrase is None
        mock_popen.return_value = self.create_mock_process(PUBLIC_CERT_CHAIN_BYTES + ENCRYPTED_EC_PRIVATE_KEY + PASSPHRASE, b'')
        cert, key, passphrase = _mtls_helper._run_cert_provider_command(['command'], expect_encrypted_key=True)
        assert cert == PUBLIC_CERT_CHAIN_BYTES
        assert key == ENCRYPTED_EC_PRIVATE_KEY
        assert passphrase == PASSPHRASE_VALUE

    @mock.patch('subprocess.Popen', autospec=True)
    def test_missing_cert(self, mock_popen):
        mock_popen.return_value = self.create_mock_process(pytest.private_key_bytes, b'')
        with pytest.raises(exceptions.ClientCertError):
            _mtls_helper._run_cert_provider_command(['command'])
        mock_popen.return_value = self.create_mock_process(ENCRYPTED_EC_PRIVATE_KEY + PASSPHRASE, b'')
        with pytest.raises(exceptions.ClientCertError):
            _mtls_helper._run_cert_provider_command(['command'], expect_encrypted_key=True)

    @mock.patch('subprocess.Popen', autospec=True)
    def test_missing_key(self, mock_popen):
        mock_popen.return_value = self.create_mock_process(pytest.public_cert_bytes, b'')
        with pytest.raises(exceptions.ClientCertError):
            _mtls_helper._run_cert_provider_command(['command'])
        mock_popen.return_value = self.create_mock_process(pytest.public_cert_bytes + PASSPHRASE, b'')
        with pytest.raises(exceptions.ClientCertError):
            _mtls_helper._run_cert_provider_command(['command'], expect_encrypted_key=True)

    @mock.patch('subprocess.Popen', autospec=True)
    def test_missing_passphrase(self, mock_popen):
        mock_popen.return_value = self.create_mock_process(pytest.public_cert_bytes + ENCRYPTED_EC_PRIVATE_KEY, b'')
        with pytest.raises(exceptions.ClientCertError):
            _mtls_helper._run_cert_provider_command(['command'], expect_encrypted_key=True)

    @mock.patch('subprocess.Popen', autospec=True)
    def test_passphrase_not_expected(self, mock_popen):
        mock_popen.return_value = self.create_mock_process(pytest.public_cert_bytes + pytest.private_key_bytes + PASSPHRASE, b'')
        with pytest.raises(exceptions.ClientCertError):
            _mtls_helper._run_cert_provider_command(['command'])

    @mock.patch('subprocess.Popen', autospec=True)
    def test_encrypted_key_expected(self, mock_popen):
        mock_popen.return_value = self.create_mock_process(pytest.public_cert_bytes + pytest.private_key_bytes + PASSPHRASE, b'')
        with pytest.raises(exceptions.ClientCertError):
            _mtls_helper._run_cert_provider_command(['command'], expect_encrypted_key=True)

    @mock.patch('subprocess.Popen', autospec=True)
    def test_unencrypted_key_expected(self, mock_popen):
        mock_popen.return_value = self.create_mock_process(pytest.public_cert_bytes + ENCRYPTED_EC_PRIVATE_KEY, b'')
        with pytest.raises(exceptions.ClientCertError):
            _mtls_helper._run_cert_provider_command(['command'])

    @mock.patch('subprocess.Popen', autospec=True)
    def test_cert_provider_returns_error(self, mock_popen):
        mock_popen.return_value = self.create_mock_process(b'', b'some error')
        mock_popen.return_value.returncode = 1
        with pytest.raises(exceptions.ClientCertError):
            _mtls_helper._run_cert_provider_command(['command'])

    @mock.patch('subprocess.Popen', autospec=True)
    def test_popen_raise_exception(self, mock_popen):
        mock_popen.side_effect = OSError()
        with pytest.raises(exceptions.ClientCertError):
            _mtls_helper._run_cert_provider_command(['command'])