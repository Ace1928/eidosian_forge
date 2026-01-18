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
def _cbReadFile(self, files, matchedFiles, directory, glob):
    if not isinstance(files, failure.Failure):
        if glob:
            glob = glob.encode('utf-8')
            matchedFiles.extend([f for f in files if fnmatch.fnmatch(f[0], glob)])
        else:
            matchedFiles.extend(files)
        d = directory.read()
        d.addBoth(self._cbReadFile, matchedFiles, directory, glob)
        return d
    else:
        reason = files
        reason.trap(EOFError)
        directory.close()
        return matchedFiles