import base64
import datetime
import ssl
from urllib.parse import urljoin, urlparse
import cryptography.hazmat.primitives.hashes
import requests
from cryptography import hazmat, x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric.dsa import DSAPublicKey
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA, EllipticCurvePublicKey
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.hashes import SHA1, Hash
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.x509 import ocsp
from redis.exceptions import AuthorizationError, ConnectionError
def _verify_response(issuer_cert, ocsp_response):
    pubkey = issuer_cert.public_key()
    try:
        if isinstance(pubkey, RSAPublicKey):
            pubkey.verify(ocsp_response.signature, ocsp_response.tbs_response_bytes, PKCS1v15(), ocsp_response.signature_hash_algorithm)
        elif isinstance(pubkey, DSAPublicKey):
            pubkey.verify(ocsp_response.signature, ocsp_response.tbs_response_bytes, ocsp_response.signature_hash_algorithm)
        elif isinstance(pubkey, EllipticCurvePublicKey):
            pubkey.verify(ocsp_response.signature, ocsp_response.tbs_response_bytes, ECDSA(ocsp_response.signature_hash_algorithm))
        else:
            pubkey.verify(ocsp_response.signature, ocsp_response.tbs_response_bytes)
    except InvalidSignature:
        raise ConnectionError('failed to valid ocsp response')