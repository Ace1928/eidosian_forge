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
def _testPermissions(uid, gid, spath, mode='r'):
    """
    checks to see if uid has proper permissions to access path with mode

    @type uid: C{int}
    @param uid: numeric user id

    @type gid: C{int}
    @param gid: numeric group id

    @type spath: C{str}
    @param spath: the path on the server to test

    @type mode: C{str}
    @param mode: 'r' or 'w' (read or write)

    @rtype: C{bool}
    @return: True if the given credentials have the specified form of
        access to the given path
    """
    if mode == 'r':
        usr = stat.S_IRUSR
        grp = stat.S_IRGRP
        oth = stat.S_IROTH
        amode = os.R_OK
    elif mode == 'w':
        usr = stat.S_IWUSR
        grp = stat.S_IWGRP
        oth = stat.S_IWOTH
        amode = os.W_OK
    else:
        raise ValueError(f"Invalid mode {mode!r}: must specify 'r' or 'w'")
    access = False
    if os.path.exists(spath):
        if uid == 0:
            access = True
        else:
            s = os.stat(spath)
            if usr & s.st_mode and uid == s.st_uid:
                access = True
            elif grp & s.st_mode and gid in _getgroups(uid):
                access = True
            elif oth & s.st_mode:
                access = True
    if access:
        if not os.access(spath, amode):
            access = False
            log.msg('Filesystem grants permission to UID %d but it is inaccessible to me running as UID %d' % (uid, os.getuid()))
    return access