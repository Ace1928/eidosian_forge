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
def _putMultipleFiles(self, files, remote):
    """
        Perform an upload for a list of local files.

        @param files: List of local files.
        @type files: C{list} of L{str}.

        @param remote: Remote path for the request relative to current working
            directory.
        @type remote: L{str}

        @return: A deferred which fires when transfer is done.
        """
    return self._cbPutMultipleNext(None, files, remote)