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
def _formatOneListResponse(self, name, size, directory, permissions, hardlinks, modified, owner, group):
    """
        Helper method to format one entry's info into a text entry like:
        'drwxrwxrwx   0 user   group   0 Jan 01  1970 filename.txt'

        @param name: C{bytes} name of the entry (file or directory or link)
        @param size: C{int} size of the entry
        @param directory: evals to C{bool} - whether the entry is a directory
        @param permissions: L{twisted.python.filepath.Permissions} object
            representing that entry's permissions
        @param hardlinks: C{int} number of hardlinks
        @param modified: C{float} - entry's last modified time in seconds
            since the epoch
        @param owner: C{str} username of the owner
        @param group: C{str} group name of the owner

        @return: C{str} in the requisite format
        """

    def formatDate(mtime):
        now = time.gmtime()
        info = {'month': _months[mtime.tm_mon], 'day': mtime.tm_mday, 'year': mtime.tm_year, 'hour': mtime.tm_hour, 'minute': mtime.tm_min}
        if now.tm_year != mtime.tm_year:
            return '%(month)s %(day)02d %(year)5d' % info
        else:
            return '%(month)s %(day)02d %(hour)02d:%(minute)02d' % info
    format = '%(directory)s%(permissions)s%(hardlinks)4d %(owner)-9s %(group)-9s %(size)15d %(date)12s '
    msg = (format % {'directory': directory and 'd' or '-', 'permissions': permissions.shorthand(), 'hardlinks': hardlinks, 'owner': owner[:8], 'group': group[:8], 'size': size, 'date': formatDate(time.gmtime(modified))}).encode(self._encoding)
    return msg + name