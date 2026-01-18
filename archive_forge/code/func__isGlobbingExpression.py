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
def _isGlobbingExpression(segments=None):
    """
    Helper for checking if a FTPShell `segments` contains a wildcard Unix
    expression.

    Only filename globbing is supported.
    This means that wildcards can only be presents in the last element of
    `segments`.

    @type  segments: C{list}
    @param segments: List of path elements as used by the FTP server protocol.

    @rtype: Boolean
    @return: True if `segments` contains a globbing expression.
    """
    if not segments:
        return False
    globCandidate = segments[-1]
    globTranslations = fnmatch.translate(globCandidate)
    nonGlobTranslations = _testTranslation.replace('TEST', globCandidate, 1)
    if nonGlobTranslations == globTranslations:
        return False
    else:
        return True