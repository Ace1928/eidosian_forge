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
def _printFailure(self, f):
    log.msg(f)
    e = f.trap(NotImplementedError, filetransfer.SFTPError, OSError, IOError)
    if e == NotImplementedError:
        self._writeToTransport(self.cmd_HELP(''))
    elif e == filetransfer.SFTPError:
        errMsg = 'remote error %i: %s\n' % (f.value.code, f.value.message)
        self._writeToTransport(errMsg)
    elif e in (OSError, IOError):
        errMsg = 'local error %i: %s\n' % (f.value.errno, f.value.strerror)
        self._writeToTransport(errMsg)