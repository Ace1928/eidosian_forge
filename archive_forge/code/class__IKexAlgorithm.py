from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error
class _IKexAlgorithm(Interface):
    """
    An L{_IKexAlgorithm} describes a key exchange algorithm.
    """
    preference = Attribute('An L{int} giving the preference of the algorithm when negotiating key exchange. Algorithms with lower precedence values are more preferred.')
    hashProcessor = Attribute('A callable hash algorithm constructor (e.g. C{hashlib.sha256}) suitable for use with this key exchange algorithm.')