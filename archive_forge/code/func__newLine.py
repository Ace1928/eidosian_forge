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
def _newLine(self):
    if self.client.transport.localClosed:
        return
    self._writeToTransport(self.ps)
    self.ignoreErrors = 0
    if self.file:
        l = self.file.readline()
        if not l:
            self.client.transport.loseConnection()
        else:
            self._writeToTransport(l)
            self.lineReceived(l.strip())