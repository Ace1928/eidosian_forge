from OpenSSL import crypto
from oauth2client_4_0 import _helpers
class OpenSSLSigner(object):
    """Signs messages with a private key."""

    def __init__(self, pkey):
        """Constructor.

        Args:
            pkey: OpenSSL.crypto.PKey (or equiv), The private key to sign with.
        """
        self._key = pkey

    def sign(self, message):
        """Signs a message.

        Args:
            message: bytes, Message to be signed.

        Returns:
            string, The signature of the message for the given key.
        """
        message = _helpers._to_bytes(message, encoding='utf-8')
        return crypto.sign(self._key, message, 'sha256')

    @staticmethod
    def from_string(key, password=b'notasecret'):
        """Construct a Signer instance from a string.

        Args:
            key: string, private key in PKCS12 or PEM format.
            password: string, password for the private key file.

        Returns:
            Signer instance.

        Raises:
            OpenSSL.crypto.Error if the key can't be parsed.
        """
        key = _helpers._to_bytes(key)
        parsed_pem_key = _helpers._parse_pem_key(key)
        if parsed_pem_key:
            pkey = crypto.load_privatekey(crypto.FILETYPE_PEM, parsed_pem_key)
        else:
            password = _helpers._to_bytes(password, encoding='utf-8')
            pkey = crypto.load_pkcs12(key, password).get_privatekey()
        return OpenSSLSigner(pkey)