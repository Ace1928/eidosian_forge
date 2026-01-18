import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
class TestDecryptPrivateKey(object):

    def test_success(self):
        decrypted_key = _mtls_helper.decrypt_private_key(ENCRYPTED_EC_PRIVATE_KEY, PASSPHRASE_VALUE)
        private_key = crypto.load_privatekey(crypto.FILETYPE_PEM, decrypted_key)
        public_key = crypto.load_publickey(crypto.FILETYPE_PEM, EC_PUBLIC_KEY)
        x509 = crypto.X509()
        x509.set_pubkey(public_key)
        signature = crypto.sign(private_key, b'data', 'sha256')
        crypto.verify(x509, signature, b'data', 'sha256')

    def test_crypto_error(self):
        with pytest.raises(crypto.Error):
            _mtls_helper.decrypt_private_key(ENCRYPTED_EC_PRIVATE_KEY, b'wrong_password')