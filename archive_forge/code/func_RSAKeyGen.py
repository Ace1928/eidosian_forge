from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.command_lib.privateca import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def RSAKeyGen(key_size=2048):
    """Generates an RSA public-private key pair.

  Args:
    key_size: The size of the RSA key, in number of bytes. Defaults to 2048.

  Returns:
    A tuple with: (private_key, public_key) both serialized in PKCS1 as bytes.
  """
    import_error_msg = 'Cannot load the Pyca cryptography library. Either the library is not installed, or site packages are not enabled for the Google Cloud SDK. Please consult Cloud KMS documentation on adding Pyca to Google Cloud SDK for further instructions.\nhttps://cloud.google.com/kms/docs/crypto'
    try:
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends.openssl.backend import backend
    except ImportError:
        log.err.Print(import_error_msg)
        sys.exit(1)
    try:
        from cryptography.hazmat.primitives.serialization.base import Encoding
        from cryptography.hazmat.primitives.serialization.base import PrivateFormat
        from cryptography.hazmat.primitives.serialization.base import PublicFormat
        from cryptography.hazmat.primitives.serialization.base import NoEncryption
    except ImportError:
        try:
            from cryptography.hazmat.primitives.serialization import Encoding
            from cryptography.hazmat.primitives.serialization import PrivateFormat
            from cryptography.hazmat.primitives.serialization import PublicFormat
            from cryptography.hazmat.primitives.serialization import NoEncryption
        except ImportError:
            log.err.Print(import_error_msg)
            sys.exit(1)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size, backend=backend)
    private_key_bytes = private_key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    public_key_bytes = private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    return (private_key_bytes, public_key_bytes)