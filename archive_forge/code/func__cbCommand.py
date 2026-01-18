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
def _cbCommand(self, result):
    if result is not None:
        if isinstance(result, str):
            result = result.encode('utf-8')
        self._writeToTransport(result)
        if not result.endswith(b'\n'):
            self._writeToTransport(b'\n')
    self._newLine()