from OpenSSL import crypto
from oauth2client_4_0 import _helpers
class OpenSSLVerifier(object):
    """Verifies the signature on a message."""

    def __init__(self, pubkey):
        """Constructor.

        Args:
            pubkey: OpenSSL.crypto.PKey, The public key to verify with.
        """
        self._pubkey = pubkey

    def verify(self, message, signature):
        """Verifies a message against a signature.

        Args:
        message: string or bytes, The message to verify. If string, will be
                 encoded to bytes as utf-8.
        signature: string or bytes, The signature on the message. If string,
                   will be encoded to bytes as utf-8.

        Returns:
            True if message was signed by the private key associated with the
            public key that this object was constructed with.
        """
        message = _helpers._to_bytes(message, encoding='utf-8')
        signature = _helpers._to_bytes(signature, encoding='utf-8')
        try:
            crypto.verify(self._pubkey, signature, message, 'sha256')
            return True
        except crypto.Error:
            return False

    @staticmethod
    def from_string(key_pem, is_x509_cert):
        """Construct a Verified instance from a string.

        Args:
            key_pem: string, public key in PEM format.
            is_x509_cert: bool, True if key_pem is an X509 cert, otherwise it
                          is expected to be an RSA key in PEM format.

        Returns:
            Verifier instance.

        Raises:
            OpenSSL.crypto.Error: if the key_pem can't be parsed.
        """
        key_pem = _helpers._to_bytes(key_pem)
        if is_x509_cert:
            pubkey = crypto.load_certificate(crypto.FILETYPE_PEM, key_pem)
        else:
            pubkey = crypto.load_privatekey(crypto.FILETYPE_PEM, key_pem)
        return OpenSSLVerifier(pubkey)