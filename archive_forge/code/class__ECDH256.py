from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
@implementer(_IEllipticCurveExchangeKexAlgorithm)
class _ECDH256:
    """
    Elliptic Curve Key Exchange with SHA-256 as HASH. Defined in
    RFC 5656.

    Note that C{ecdh-sha2-nistp256} takes priority over nistp384 or nistp512.
    This is the same priority from OpenSSH.

    C{ecdh-sha2-nistp256} is considered preety good cryptography.
    If you need something better consider using C{curve25519-sha256}.
    """
    preference = 3
    hashProcessor = sha256