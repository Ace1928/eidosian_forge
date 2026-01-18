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
@implementer(ISSHPrivateKey)
class SSHPrivateKey:

    def __init__(self, username, algName, blob, sigData, signature):
        self.username = username
        self.algName = algName
        self.blob = blob
        self.sigData = sigData
        self.signature = signature