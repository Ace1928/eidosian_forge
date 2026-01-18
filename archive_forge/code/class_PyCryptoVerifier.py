from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto.Util.asn1 import DerSequence
from oauth2client_4_0 import _helpers
class PyCryptoVerifier(object):
    """Verifies the signature on a message."""

    def __init__(self, pubkey):
        """Constructor.

        Args:
            pubkey: OpenSSL.crypto.PKey (or equiv), The public key to verify
            with.
        """
        self._pubkey = pubkey

    def verify(self, message, signature):
        """Verifies a message against a signature.

        Args:
            message: string or bytes, The message to verify. If string, will be
                     encoded to bytes as utf-8.
            signature: string or bytes, The signature on the message.

        Returns:
            True if message was signed by the private key associated with the
            public key that this object was constructed with.
        """
        message = _helpers._to_bytes(message, encoding='utf-8')
        return PKCS1_v1_5.new(self._pubkey).verify(SHA256.new(message), signature)

    @staticmethod
    def from_string(key_pem, is_x509_cert):
        """Construct a Verified instance from a string.

        Args:
            key_pem: string, public key in PEM format.
            is_x509_cert: bool, True if key_pem is an X509 cert, otherwise it
                          is expected to be an RSA key in PEM format.

        Returns:
            Verifier instance.
        """
        if is_x509_cert:
            key_pem = _helpers._to_bytes(key_pem)
            pemLines = key_pem.replace(b' ', b'').split()
            certDer = _helpers._urlsafe_b64decode(b''.join(pemLines[1:-1]))
            certSeq = DerSequence()
            certSeq.decode(certDer)
            tbsSeq = DerSequence()
            tbsSeq.decode(certSeq[0])
            pubkey = RSA.importKey(tbsSeq[6])
        else:
            pubkey = RSA.importKey(key_pem)
        return PyCryptoVerifier(pubkey)