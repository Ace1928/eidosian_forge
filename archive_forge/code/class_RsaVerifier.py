from pyasn1.codec.der import decoder
from pyasn1_modules import pem
from pyasn1_modules.rfc2459 import Certificate
from pyasn1_modules.rfc5208 import PrivateKeyInfo
import rsa
import six
from oauth2client import _helpers
class RsaVerifier(object):
    """Verifies the signature on a message.

    Args:
        pubkey: rsa.key.PublicKey (or equiv), The public key to verify with.
    """

    def __init__(self, pubkey):
        self._pubkey = pubkey

    def verify(self, message, signature):
        """Verifies a message against a signature.

        Args:
            message: string or bytes, The message to verify. If string, will be
                     encoded to bytes as utf-8.
            signature: string or bytes, The signature on the message. If
                       string, will be encoded to bytes as utf-8.

        Returns:
            True if message was signed by the private key associated with the
            public key that this object was constructed with.
        """
        message = _helpers._to_bytes(message, encoding='utf-8')
        try:
            return rsa.pkcs1.verify(message, signature, self._pubkey)
        except (ValueError, rsa.pkcs1.VerificationError):
            return False

    @classmethod
    def from_string(cls, key_pem, is_x509_cert):
        """Construct an RsaVerifier instance from a string.

        Args:
            key_pem: string, public key in PEM format.
            is_x509_cert: bool, True if key_pem is an X509 cert, otherwise it
                          is expected to be an RSA key in PEM format.

        Returns:
            RsaVerifier instance.

        Raises:
            ValueError: if the key_pem can't be parsed. In either case, error
                        will begin with 'No PEM start marker'. If
                        ``is_x509_cert`` is True, will fail to find the
                        "-----BEGIN CERTIFICATE-----" error, otherwise fails
                        to find "-----BEGIN RSA PUBLIC KEY-----".
        """
        key_pem = _helpers._to_bytes(key_pem)
        if is_x509_cert:
            der = rsa.pem.load_pem(key_pem, 'CERTIFICATE')
            asn1_cert, remaining = decoder.decode(der, asn1Spec=Certificate())
            if remaining != b'':
                raise ValueError('Unused bytes', remaining)
            cert_info = asn1_cert['tbsCertificate']['subjectPublicKeyInfo']
            key_bytes = _bit_list_to_bytes(cert_info['subjectPublicKey'])
            pubkey = rsa.PublicKey.load_pkcs1(key_bytes, 'DER')
        else:
            pubkey = rsa.PublicKey.load_pkcs1(key_pem, 'PEM')
        return cls(pubkey)