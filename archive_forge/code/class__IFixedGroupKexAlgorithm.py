from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
class _IFixedGroupKexAlgorithm(_IKexAlgorithm):
    """
    An L{_IFixedGroupKexAlgorithm} describes a key exchange algorithm with a
    fixed prime / generator group.
    """
    prime = Attribute('An L{int} giving the prime number used in Diffie-Hellman key exchange, or L{None} if not applicable.')
    generator = Attribute('An L{int} giving the generator number used in Diffie-Hellman key exchange, or L{None} if not applicable. (This is not related to Python generator functions.)')