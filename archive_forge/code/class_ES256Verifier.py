from cryptography import utils  # type: ignore
import cryptography.exceptions
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
import cryptography.x509
from google.auth import _helpers
from google.auth.crypt import base
class ES256Verifier(base.Verifier):
    """Verifies ECDSA cryptographic signatures using public keys.

    Args:
        public_key (
                cryptography.hazmat.primitives.asymmetric.ec.ECDSAPublicKey):
            The public key used to verify signatures.
    """

    def __init__(self, public_key):
        self._pubkey = public_key

    @_helpers.copy_docstring(base.Verifier)
    def verify(self, message, signature):
        sig_bytes = _helpers.to_bytes(signature)
        if len(sig_bytes) != 64:
            return False
        r = int.from_bytes(sig_bytes[:32], byteorder='big') if _helpers.is_python_3() else utils.int_from_bytes(sig_bytes[:32], byteorder='big')
        s = int.from_bytes(sig_bytes[32:], byteorder='big') if _helpers.is_python_3() else utils.int_from_bytes(sig_bytes[32:], byteorder='big')
        asn1_sig = encode_dss_signature(r, s)
        message = _helpers.to_bytes(message)
        try:
            self._pubkey.verify(asn1_sig, message, ec.ECDSA(hashes.SHA256()))
            return True
        except (ValueError, cryptography.exceptions.InvalidSignature):
            return False

    @classmethod
    def from_string(cls, public_key):
        """Construct an Verifier instance from a public key or public
        certificate string.

        Args:
            public_key (Union[str, bytes]): The public key in PEM format or the
                x509 public key certificate.

        Returns:
            Verifier: The constructed verifier.

        Raises:
            ValueError: If the public key can't be parsed.
        """
        public_key_data = _helpers.to_bytes(public_key)
        if _CERTIFICATE_MARKER in public_key_data:
            cert = cryptography.x509.load_pem_x509_certificate(public_key_data, _BACKEND)
            pubkey = cert.public_key()
        else:
            pubkey = serialization.load_pem_public_key(public_key_data, _BACKEND)
        return cls(pubkey)