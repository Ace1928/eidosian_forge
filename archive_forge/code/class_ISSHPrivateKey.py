import base64
import hmac
import random
import re
import time
from binascii import hexlify
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred import error
from twisted.cred._digest import calcHA1, calcHA2, calcResponse
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.randbytes import secureRandom
from twisted.python.versions import Version
class ISSHPrivateKey(ICredentials):
    """
    L{ISSHPrivateKey} credentials encapsulate an SSH public key to be checked
    against a user's private key.

    @ivar username: The username associated with these credentials.
    @type username: L{bytes}

    @ivar algName: The algorithm name for the blob.
    @type algName: L{bytes}

    @ivar blob: The public key blob as sent by the client.
    @type blob: L{bytes}

    @ivar sigData: The data the signature was made from.
    @type sigData: L{bytes}

    @ivar signature: The signed data.  This is checked to verify that the user
        owns the private key.
    @type signature: L{bytes} or L{None}
    """