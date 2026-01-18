import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
class TestGetClientCertAndKey(object):

    def test_callback_success(self):
        callback = mock.Mock()
        callback.return_value = (pytest.public_cert_bytes, pytest.private_key_bytes)
        found_cert_key, cert, key = _mtls_helper.get_client_cert_and_key(callback)
        assert found_cert_key
        assert cert == pytest.public_cert_bytes
        assert key == pytest.private_key_bytes

    @mock.patch('google.auth.transport._mtls_helper.get_client_ssl_credentials', autospec=True)
    def test_use_metadata(self, mock_get_client_ssl_credentials):
        mock_get_client_ssl_credentials.return_value = (True, pytest.public_cert_bytes, pytest.private_key_bytes, None)
        found_cert_key, cert, key = _mtls_helper.get_client_cert_and_key()
        assert found_cert_key
        assert cert == pytest.public_cert_bytes
        assert key == pytest.private_key_bytes