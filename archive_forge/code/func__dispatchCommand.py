import fcntl
import fnmatch
import getpass
import glob
import os
import pwd
import stat
import struct
import sys
import tty
from typing import List, Optional, TextIO, Union
from twisted.conch.client import connect, default, options
from twisted.conch.ssh import channel, common, connection, filetransfer
from twisted.internet import defer, reactor, stdio, utils
from twisted.protocols import basic
from twisted.python import failure, log, usage
from twisted.python.filepath import FilePath
def _dispatchCommand(self, line):
    if ' ' in line:
        command, rest = line.split(' ', 1)
        rest = rest.lstrip()
    else:
        command, rest = (line, '')
    if command.startswith('!'):
        f = self.cmd_EXEC
        rest = (command[1:] + ' ' + rest).strip()
    else:
        command = command.upper()
        log.msg('looking up cmd %s' % command)
        f = getattr(self, 'cmd_%s' % command, None)
    if f is not None:
        return defer.maybeDeferred(f, rest)
    else:
        errMsg = "No command called `%s'" % command
        self._ebCommand(failure.Failure(NotImplementedError(errMsg)))
        self._newLine()