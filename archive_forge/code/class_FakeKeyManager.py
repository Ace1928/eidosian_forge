import base64
import datetime
import mock
from castellan.common.exception import KeyManagerError
from castellan.common.exception import ManagedObjectNotFoundError
import cryptography.exceptions as crypto_exceptions
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from oslo_utils import timeutils
from cursive import exception
from cursive import signature_utils
from cursive.tests import base
class FakeKeyManager(object):

    def __init__(self):
        self.certs = {'invalid_format_cert': FakeCastellanCertificate('A' * 256, 'BLAH'), 'valid_format_cert': FakeCastellanCertificate('A' * 256, 'X.509'), 'invalid-cert-uuid': ManagedObjectNotFoundError()}

    def get(self, context, cert_uuid):
        cert = self.certs.get(cert_uuid)
        if cert is None:
            raise KeyManagerError('No matching certificate found.')
        if isinstance(cert, ManagedObjectNotFoundError):
            raise cert
        return cert