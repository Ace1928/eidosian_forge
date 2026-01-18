import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
class DccFileReceive(DccFileReceiveBasic):
    """
    Higher-level coverage for getting a file from DCC SEND.

    I allow you to change the file's name and destination directory.  I won't
    overwrite an existing file unless I've been told it's okay to do so.  If
    passed the resumeOffset keyword argument I will attempt to resume the file
    from that amount of bytes.

    XXX: I need to let the client know when I am finished.
    XXX: I need to decide how to keep a progress indicator updated.
    XXX: Client needs a way to tell me "Do not finish until I say so."
    XXX: I need to make sure the client understands if the file cannot be written.

    @ivar filename: The name of the file to get.
    @type filename: L{bytes}

    @ivar fileSize: The size of the file to get, which has a default value of
        C{-1} if the size of the file was not specified in the DCC SEND
        request.
    @type fileSize: L{int}

    @ivar destDir: The destination directory for the file to be received.
    @type destDir: L{bytes}

    @ivar overwrite: An integer representing whether an existing file should be
        overwritten or not.  This initially is an L{int} but can be modified to
        be a L{bool} using the L{set_overwrite} method.
    @type overwrite: L{int} or L{bool}

    @ivar queryData: queryData is a 3-tuple of (user, channel, data).
    @type queryData: L{tuple}

    @ivar fromUser: This is the hostmask of the requesting user and is found at
        index 0 of L{queryData}.
    @type fromUser: L{bytes}
    """
    filename = 'dcc'
    fileSize = -1
    destDir = '.'
    overwrite = 0
    fromUser: Optional[bytes] = None
    queryData = None

    def __init__(self, filename, fileSize=-1, queryData=None, destDir='.', resumeOffset=0):
        DccFileReceiveBasic.__init__(self, resumeOffset=resumeOffset)
        self.filename = filename
        self.destDir = destDir
        self.fileSize = fileSize
        self._resumeOffset = resumeOffset
        if queryData:
            self.queryData = queryData
            self.fromUser = self.queryData[0]

    def set_directory(self, directory):
        """
        Set the directory where the downloaded file will be placed.

        May raise OSError if the supplied directory path is not suitable.

        @param directory: The directory where the file to be received will be
            placed.
        @type directory: L{bytes}
        """
        if not path.exists(directory):
            raise OSError(errno.ENOENT, 'You see no directory there.', directory)
        if not path.isdir(directory):
            raise OSError(errno.ENOTDIR, 'You cannot put a file into something which is not a directory.', directory)
        if not os.access(directory, os.X_OK | os.W_OK):
            raise OSError(errno.EACCES, 'This directory is too hard to write in to.', directory)
        self.destDir = directory

    def set_filename(self, filename):
        """
        Change the name of the file being transferred.

        This replaces the file name provided by the sender.

        @param filename: The new name for the file.
        @type filename: L{bytes}
        """
        self.filename = filename

    def set_overwrite(self, boolean):
        """
        May I overwrite existing files?

        @param boolean: A boolean value representing whether existing files
            should be overwritten or not.
        @type boolean: L{bool}
        """
        self.overwrite = boolean

    def connectionMade(self):
        dst = path.abspath(path.join(self.destDir, self.filename))
        exists = path.exists(dst)
        if self.resume and exists:
            self.file = open(dst, 'rb+')
            self.file.seek(self._resumeOffset)
            self.file.truncate()
            log.msg('Attempting to resume %s - starting from %d bytes' % (self.file, self.file.tell()))
        elif self.resume and (not exists):
            raise OSError(errno.ENOENT, 'You cannot resume writing to a file that does not exist!', dst)
        elif self.overwrite or not exists:
            self.file = open(dst, 'wb')
        else:
            raise OSError(errno.EEXIST, "There's a file in the way.  Perhaps that's why you cannot open it.", dst)

    def dataReceived(self, data):
        self.file.write(data)
        DccFileReceiveBasic.dataReceived(self, data)

    def connectionLost(self, reason):
        """
        When the connection is lost, I close the file.

        @param reason: The reason why the connection was lost.
        @type reason: L{Failure}
        """
        self.connected = 0
        logmsg = f'{self} closed.'
        if self.fileSize > 0:
            logmsg = '%s  %d/%d bytes received' % (logmsg, self.bytesReceived, self.fileSize)
            if self.bytesReceived == self.fileSize:
                pass
            elif self.bytesReceived < self.fileSize:
                logmsg = '%s (Warning: %d bytes short)' % (logmsg, self.fileSize - self.bytesReceived)
            else:
                logmsg = f'{logmsg} (file larger than expected)'
        else:
            logmsg = '%s  %d bytes received' % (logmsg, self.bytesReceived)
        if hasattr(self, 'file'):
            logmsg = f'{logmsg} and written to {self.file.name}.\n'
            if hasattr(self.file, 'close'):
                self.file.close()

    def __str__(self) -> str:
        if not self.connected:
            return f'<Unconnected DccFileReceive object at {id(self):x}>'
        transport = self.transport
        assert transport is not None
        from_ = str(transport.getPeer())
        if self.fromUser is not None:
            from_ = f'{self.fromUser!r} ({from_})'
        s = f"DCC transfer of '{self.filename}' from {from_}"
        return s

    def __repr__(self) -> str:
        s = f'<{self.__class__} at {id(self):x}: GET {self.filename}>'
        return s