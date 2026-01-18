from pyasn1.codec.der import decoder
from pyasn1_modules import pem
from pyasn1_modules.rfc2459 import Certificate
from pyasn1_modules.rfc5208 import PrivateKeyInfo
import rsa
import six
from oauth2client import _helpers
Construct an RsaSigner instance from a string.

        Args:
            key: string, private key in PEM format.
            password: string, password for private key file. Unused for PEM
                      files.

        Returns:
            RsaSigner instance.

        Raises:
            ValueError if the key cannot be parsed as PKCS#1 or PKCS#8 in
            PEM format.
        