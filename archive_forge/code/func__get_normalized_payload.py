import calendar
import time
import urllib
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization
from cryptography import x509 as cryptography_x509
from keystoneauth1 import identity
from keystoneauth1 import loading
from keystoneauth1 import service_token
from keystoneauth1 import session
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from castellan.common import exception
from castellan.common.objects import key as key_base_class
from castellan.common.objects import opaque_data as op_data
from castellan.i18n import _
from castellan.key_manager import key_manager
from barbicanclient import client as barbican_client_import
from barbicanclient import exceptions as barbican_exceptions
from oslo_utils import timeutils
def _get_normalized_payload(self, encoded_bytes, secret_type):
    """Normalizes the bytes of the object.

        Barbican expects certificates, public keys, and private keys in PEM
        format, but Castellan expects these objects to be DER encoded bytes
        instead.
        """
    if secret_type == 'public':
        key = serialization.load_der_public_key(encoded_bytes, backend=backends.default_backend())
        return key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)
    elif secret_type == 'private':
        key = serialization.load_der_private_key(encoded_bytes, backend=backends.default_backend(), password=None)
        return key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption())
    elif secret_type == 'certificate':
        cert = cryptography_x509.load_der_x509_certificate(encoded_bytes, backend=backends.default_backend())
        return cert.public_bytes(encoding=serialization.Encoding.PEM)
    else:
        return encoded_bytes