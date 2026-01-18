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
def _cbPutMultipleNext(self, previousResult, files, remotePath, single=False):
    """
        Perform an upload for the next file in the list of local files.

        @param previousResult: Result form previous file form the list.
        @type previousResult: L{str}

        @param files: List of local files.
        @type files: C{list} of L{str}

        @param remotePath: Remote path for the request relative to current
            working directory.
        @type remotePath: L{str}

        @param single: A flag which signals if this is a transfer for a single
            file in which case we use the exact remote path
        @type single: L{bool}

        @return: A deferred which fires when transfer is done.
        """
    if isinstance(previousResult, failure.Failure):
        self._printFailure(previousResult)
    elif previousResult:
        if isinstance(previousResult, str):
            previousResult = previousResult.encode('utf-8')
        self._writeToTransport(previousResult)
        if not previousResult.endswith(b'\n'):
            self._writeToTransport(b'\n')
    currentFile = None
    while files and (not currentFile):
        try:
            currentFile = files.pop(0)
            localStream = open(currentFile, 'rb')
        except BaseException:
            self._printFailure(failure.Failure())
            currentFile = None
    if not currentFile:
        return None
    if single:
        remote = remotePath
    else:
        name = os.path.split(currentFile)[1]
        remote = os.path.join(remotePath, name)
        log.msg((name, remote, remotePath))
    d = self._putRemoteFile(localStream, remote)
    d.addBoth(self._cbPutMultipleNext, files, remotePath)
    return d