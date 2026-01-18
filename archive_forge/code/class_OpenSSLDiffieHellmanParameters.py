from __future__ import annotations
import warnings
from binascii import hexlify
from functools import lru_cache
from hashlib import md5
from typing import Dict
from zope.interface import Interface, implementer
from OpenSSL import SSL, crypto
from OpenSSL._util import lib as pyOpenSSLlib
import attr
from constantly import FlagConstant, Flags, NamedConstant, Names
from incremental import Version
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.error import CertificateError, VerifyError
from twisted.internet.interfaces import (
from twisted.python import log, util
from twisted.python.compat import nativeString
from twisted.python.deprecate import _mutuallyExclusiveArguments, deprecated
from twisted.python.failure import Failure
from twisted.python.randbytes import secureRandom
from ._idna import _idnaBytes
class OpenSSLDiffieHellmanParameters:
    """
    A representation of key generation parameters that are required for
    Diffie-Hellman key exchange.
    """

    def __init__(self, parameters):
        self._dhFile = parameters

    @classmethod
    def fromFile(cls, filePath):
        """
        Load parameters from a file.

        Such a file can be generated using the C{openssl} command line tool as
        following:

        C{openssl dhparam -out dh_param_2048.pem -2 2048}

        Please refer to U{OpenSSL's C{dhparam} documentation
        <http://www.openssl.org/docs/apps/dhparam.html>} for further details.

        @param filePath: A file containing parameters for Diffie-Hellman key
            exchange.
        @type filePath: L{FilePath <twisted.python.filepath.FilePath>}

        @return: An instance that loads its parameters from C{filePath}.
        @rtype: L{DiffieHellmanParameters
            <twisted.internet.ssl.DiffieHellmanParameters>}
        """
        return cls(filePath)