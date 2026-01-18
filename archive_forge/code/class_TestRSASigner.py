import json
import os
from cryptography.hazmat.primitives.asymmetric import rsa
import pytest  # type: ignore
from google.auth import _helpers
from google.auth.crypt import _cryptography_rsa
from google.auth.crypt import base
class TestRSASigner(object):

    def test_from_string_pkcs1(self):
        signer = _cryptography_rsa.RSASigner.from_string(PKCS1_KEY_BYTES)
        assert isinstance(signer, _cryptography_rsa.RSASigner)
        assert isinstance(signer._key, rsa.RSAPrivateKey)

    def test_from_string_pkcs1_unicode(self):
        key_bytes = _helpers.from_bytes(PKCS1_KEY_BYTES)
        signer = _cryptography_rsa.RSASigner.from_string(key_bytes)
        assert isinstance(signer, _cryptography_rsa.RSASigner)
        assert isinstance(signer._key, rsa.RSAPrivateKey)

    def test_from_string_pkcs8(self):
        signer = _cryptography_rsa.RSASigner.from_string(PKCS8_KEY_BYTES)
        assert isinstance(signer, _cryptography_rsa.RSASigner)
        assert isinstance(signer._key, rsa.RSAPrivateKey)

    def test_from_string_pkcs8_unicode(self):
        key_bytes = _helpers.from_bytes(PKCS8_KEY_BYTES)
        signer = _cryptography_rsa.RSASigner.from_string(key_bytes)
        assert isinstance(signer, _cryptography_rsa.RSASigner)
        assert isinstance(signer._key, rsa.RSAPrivateKey)

    def test_from_string_pkcs12(self):
        with pytest.raises(ValueError):
            _cryptography_rsa.RSASigner.from_string(PKCS12_KEY_BYTES)

    def test_from_string_bogus_key(self):
        key_bytes = 'bogus-key'
        with pytest.raises(ValueError):
            _cryptography_rsa.RSASigner.from_string(key_bytes)

    def test_from_service_account_info(self):
        signer = _cryptography_rsa.RSASigner.from_service_account_info(SERVICE_ACCOUNT_INFO)
        assert signer.key_id == SERVICE_ACCOUNT_INFO[base._JSON_FILE_PRIVATE_KEY_ID]
        assert isinstance(signer._key, rsa.RSAPrivateKey)

    def test_from_service_account_info_missing_key(self):
        with pytest.raises(ValueError) as excinfo:
            _cryptography_rsa.RSASigner.from_service_account_info({})
        assert excinfo.match(base._JSON_FILE_PRIVATE_KEY)

    def test_from_service_account_file(self):
        signer = _cryptography_rsa.RSASigner.from_service_account_file(SERVICE_ACCOUNT_JSON_FILE)
        assert signer.key_id == SERVICE_ACCOUNT_INFO[base._JSON_FILE_PRIVATE_KEY_ID]
        assert isinstance(signer._key, rsa.RSAPrivateKey)