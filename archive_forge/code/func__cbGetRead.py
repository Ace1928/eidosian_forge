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
def _cbGetRead(self, data, rf, lf, chunks, start, size, startTime):
    if data and isinstance(data, failure.Failure):
        log.msg('get read err: %s' % data)
        reason = data
        reason.trap(EOFError)
        i = chunks.index((start, start + size))
        del chunks[i]
        chunks.insert(i, (start, 'eof'))
    elif data:
        log.msg('get read data: %i' % len(data))
        lf.seek(start)
        lf.write(data)
        if len(data) != size:
            log.msg('got less than we asked for: %i < %i' % (len(data), size))
            i = chunks.index((start, start + size))
            del chunks[i]
            chunks.insert(i, (start, start + len(data)))
        rf.total += len(data)
    if self.useProgressBar:
        self._printProgressBar(rf, startTime)
    chunk = self._getNextChunk(chunks)
    if not chunk:
        return
    else:
        start, length = chunk
    log.msg('asking for %i -> %i' % (start, start + length))
    d = rf.readChunk(start, length)
    d.addBoth(self._cbGetRead, rf, lf, chunks, start, length, startTime)
    return d