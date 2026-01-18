from OpenSSL import crypto
from oauth2client_4_0 import _helpers
Construct a Signer instance from a string.

        Args:
            key: string, private key in PKCS12 or PEM format.
            password: string, password for the private key file.

        Returns:
            Signer instance.

        Raises:
            OpenSSL.crypto.Error if the key can't be parsed.
        