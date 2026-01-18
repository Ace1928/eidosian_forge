from __future__ import annotations
import email.base64mime
import email.generator
import email.message
import email.policy
import io
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import pkcs7 as rust_pkcs7
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.utils import _check_byteslike
class PKCS7SignatureBuilder:

    def __init__(self, data: typing.Optional[bytes]=None, signers: typing.List[typing.Tuple[x509.Certificate, PKCS7PrivateKeyTypes, PKCS7HashTypes]]=[], additional_certs: typing.List[x509.Certificate]=[]):
        self._data = data
        self._signers = signers
        self._additional_certs = additional_certs

    def set_data(self, data: bytes) -> PKCS7SignatureBuilder:
        _check_byteslike('data', data)
        if self._data is not None:
            raise ValueError('data may only be set once')
        return PKCS7SignatureBuilder(data, self._signers)

    def add_signer(self, certificate: x509.Certificate, private_key: PKCS7PrivateKeyTypes, hash_algorithm: PKCS7HashTypes) -> PKCS7SignatureBuilder:
        if not isinstance(hash_algorithm, (hashes.SHA224, hashes.SHA256, hashes.SHA384, hashes.SHA512)):
            raise TypeError('hash_algorithm must be one of hashes.SHA224, SHA256, SHA384, or SHA512')
        if not isinstance(certificate, x509.Certificate):
            raise TypeError('certificate must be a x509.Certificate')
        if not isinstance(private_key, (rsa.RSAPrivateKey, ec.EllipticCurvePrivateKey)):
            raise TypeError('Only RSA & EC keys are supported at this time.')
        return PKCS7SignatureBuilder(self._data, self._signers + [(certificate, private_key, hash_algorithm)])

    def add_certificate(self, certificate: x509.Certificate) -> PKCS7SignatureBuilder:
        if not isinstance(certificate, x509.Certificate):
            raise TypeError('certificate must be a x509.Certificate')
        return PKCS7SignatureBuilder(self._data, self._signers, self._additional_certs + [certificate])

    def sign(self, encoding: serialization.Encoding, options: typing.Iterable[PKCS7Options], backend: typing.Any=None) -> bytes:
        if len(self._signers) == 0:
            raise ValueError('Must have at least one signer')
        if self._data is None:
            raise ValueError('You must add data to sign')
        options = list(options)
        if not all((isinstance(x, PKCS7Options) for x in options)):
            raise ValueError('options must be from the PKCS7Options enum')
        if encoding not in (serialization.Encoding.PEM, serialization.Encoding.DER, serialization.Encoding.SMIME):
            raise ValueError('Must be PEM, DER, or SMIME from the Encoding enum')
        if PKCS7Options.Text in options and PKCS7Options.DetachedSignature not in options:
            raise ValueError('When passing the Text option you must also pass DetachedSignature')
        if PKCS7Options.Text in options and encoding in (serialization.Encoding.DER, serialization.Encoding.PEM):
            raise ValueError('The Text option is only available for SMIME serialization')
        if PKCS7Options.NoAttributes in options and PKCS7Options.NoCapabilities in options:
            raise ValueError('NoAttributes is a superset of NoCapabilities. Do not pass both values.')
        return rust_pkcs7.sign_and_serialize(self, encoding, options)