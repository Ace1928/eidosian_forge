from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
class _IGroupExchangeKexAlgorithm(_IKexAlgorithm):
    """
    An L{_IGroupExchangeKexAlgorithm} describes a key exchange algorithm
    that uses group exchange between the client and server.

    A prime / generator group should be chosen at run time based on the
    requested size. See RFC 4419.
    """