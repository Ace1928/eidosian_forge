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
def _tolerateErrors(wrapped):
    """
    Wrap up an C{info_callback} for pyOpenSSL so that if something goes wrong
    the error is immediately logged and the connection is dropped if possible.

    This wrapper exists because some versions of pyOpenSSL don't handle errors
    from callbacks at I{all}, and those which do write tracebacks directly to
    stderr rather than to a supplied logging system.  This reports unexpected
    errors to the Twisted logging system.

    Also, this terminates the connection immediately if possible because if
    you've got bugs in your verification logic it's much safer to just give up.

    @param wrapped: A valid C{info_callback} for pyOpenSSL.
    @type wrapped: L{callable}

    @return: A valid C{info_callback} for pyOpenSSL that handles any errors in
        C{wrapped}.
    @rtype: L{callable}
    """

    def infoCallback(connection, where, ret):
        try:
            return wrapped(connection, where, ret)
        except BaseException:
            f = Failure()
            log.err(f, 'Error during info_callback')
            connection.get_app_data().failVerification(f)
    return infoCallback