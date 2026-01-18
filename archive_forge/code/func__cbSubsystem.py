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
def _cbSubsystem(self, result):
    self.client = filetransfer.FileTransferClient()
    self.client.makeConnection(self)
    self.dataReceived = self.client.dataReceived
    f = None
    if self.conn.options['batchfile']:
        fn = self.conn.options['batchfile']
        if fn != '-':
            f = open(fn)
    self.stdio = stdio.StandardIO(StdioClient(self.client, f))