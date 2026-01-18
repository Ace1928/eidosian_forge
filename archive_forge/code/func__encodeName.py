import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
def _encodeName(self, name):
    """
        Encode C{name} to be sent over the wire.

        This encodes L{unicode} objects as UTF-8 and leaves L{bytes} as-is.

        As described by U{RFC 3659 section
        2.2<https://tools.ietf.org/html/rfc3659#section-2.2>}::

            Various FTP commands take pathnames as arguments, or return
            pathnames in responses. When the MLST command is supported, as
            indicated in the response to the FEAT command, pathnames are to be
            transferred in one of the following two formats.

                pathname = utf-8-name / raw
                utf-8-name = <a UTF-8 encoded Unicode string>
                raw = <any string that is not a valid UTF-8 encoding>

            Which format is used is at the option of the user-PI or server-PI
            sending the pathname.

        @param name: Name to be encoded.
        @type name: L{bytes} or L{unicode}

        @return: Wire format of C{name}.
        @rtype: L{bytes}
        """
    if isinstance(name, str):
        return name.encode('utf-8')
    return name