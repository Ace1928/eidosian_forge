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
def _cbGetMultipleNext(self, res, files, local):
    if isinstance(res, failure.Failure):
        self._printFailure(res)
    elif res:
        self._writeToTransport(res)
        if not res.endswith('\n'):
            self._writeToTransport('\n')
    if not files:
        return
    f = files.pop(0)[0]
    lf = open(os.path.join(local, os.path.split(f)[1]), 'wb', 0)
    path = FilePath(self.currentDirectory).child(f)
    d = self.client.openFile(path.path, filetransfer.FXF_READ, {})
    d.addCallback(self._cbGetOpenFile, lf)
    d.addErrback(self._ebCloseLf, lf)
    d.addBoth(self._cbGetMultipleNext, files, local)
    return d