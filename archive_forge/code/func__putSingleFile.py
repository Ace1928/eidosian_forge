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
def _putSingleFile(self, local, remote):
    """
        Perform an upload for a single file.

        @param local: Path to local file.
        @type local: L{str}.

        @param remote: Remote path for the request relative to current working
            directory.
        @type remote: L{str}

        @return: A deferred which fires when transfer is done.
        """
    return self._cbPutMultipleNext(None, [local], remote, single=True)