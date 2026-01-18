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
def _cbPutWrite(self, ignored, rf, lf, chunks, startTime):
    chunk = self._getNextChunk(chunks)
    start, size = chunk
    lf.seek(start)
    data = lf.read(size)
    if self.useProgressBar:
        lf.total += len(data)
        self._printProgressBar(lf, startTime)
    if data:
        d = rf.writeChunk(start, data)
        d.addCallback(self._cbPutWrite, rf, lf, chunks, startTime)
        return d
    else:
        return