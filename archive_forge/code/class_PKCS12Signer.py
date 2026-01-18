from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from google.auth import _helpers
from google.auth.crypt import base as crypt_base
from google.oauth2 import service_account
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
class PKCS12Signer(crypt_base.Signer, crypt_base.FromServiceAccountMixin):
    """Signer for a p12 service account key based on pyca/cryptography."""

    def __init__(self, key):
        self._key = key

    @property
    def key_id(self):
        return None

    def sign(self, message):
        message = _helpers.to_bytes(message)
        from google.auth.crypt import _cryptography_rsa
        return self._key.sign(message, _cryptography_rsa._PADDING, _cryptography_rsa._SHA256)

    @classmethod
    def from_string(cls, key_strings, key_id=None):
        del key_id
        key_string, password = (_helpers.to_bytes(k) for k in key_strings)
        from cryptography.hazmat.primitives.serialization import pkcs12
        from cryptography.hazmat import backends
        key, _, _ = pkcs12.load_key_and_certificates(key_string, password, backend=backends.default_backend())
        return cls(key)