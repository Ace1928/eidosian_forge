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
def _getgroups(uid):
    """
    Return the primary and supplementary groups for the given UID.

    @type uid: C{int}
    """
    result = []
    pwent = pwd.getpwuid(uid)
    result.append(pwent.pw_gid)
    for grent in grp.getgrall():
        if pwent.pw_name in grent.gr_mem:
            result.append(grent.gr_gid)
    return result