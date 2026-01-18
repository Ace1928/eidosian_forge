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
def _abbrevTime(self, t):
    if t > 3600:
        hours = int(t / 3600)
        t -= 3600 * hours
        mins = int(t / 60)
        t -= 60 * mins
        return '%i:%02i:%02i' % (hours, mins, t)
    else:
        mins = int(t / 60)
        t -= 60 * mins
        return '%02i:%02i' % (mins, t)