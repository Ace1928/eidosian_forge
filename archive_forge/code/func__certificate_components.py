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
def _certificate_components(self, cert):
    """Given an SSL certificate, retract the useful components for
        validating the certificate status with an OCSP server.

        Args:
            cert ([bytes]): A PEM encoded ssl certificate
        """
    try:
        aia = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.AUTHORITY_INFORMATION_ACCESS).value
    except cryptography.x509.extensions.ExtensionNotFound:
        raise ConnectionError('No AIA information present in ssl certificate')
    issuers = [i for i in aia if i.access_method == x509.oid.AuthorityInformationAccessOID.CA_ISSUERS]
    try:
        issuer = issuers[0].access_location.value
    except IndexError:
        issuer = None
    ocsps = [i for i in aia if i.access_method == x509.oid.AuthorityInformationAccessOID.OCSP]
    try:
        ocsp = ocsps[0].access_location.value
    except IndexError:
        raise ConnectionError('no ocsp servers in certificate')
    return (cert, issuer, ocsp)