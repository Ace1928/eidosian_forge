from hashlib import sha1, sha256, sha384, sha512
from zope.interface import Attribute, Interface, implementer
from twisted.conch import error

    Get a list of supported key exchange algorithm names in order of
    preference.

    @return: A C{list} of supported key exchange algorithm names.
    @rtype: C{list} of L{bytes}
    