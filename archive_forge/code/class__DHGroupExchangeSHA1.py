from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
@implementer(_IGroupExchangeKexAlgorithm)
class _DHGroupExchangeSHA1:
    """
    Diffie-Hellman Group and Key Exchange with SHA-1 as HASH. Defined in
    RFC 4419, 4.1.
    """
    preference = 7
    hashProcessor = sha1