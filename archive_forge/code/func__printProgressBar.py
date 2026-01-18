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
def _printProgressBar(self, f, startTime):
    """
        Update a console progress bar on this L{StdioClient}'s transport, based
        on the difference between the start time of the operation and the
        current time according to the reactor, and appropriate to the size of
        the console window.

        @param f: a wrapper around the file which is being written or read
        @type f: L{FileWrapper}

        @param startTime: The time at which the operation being tracked began.
        @type startTime: L{float}
        """
    diff = self.reactor.seconds() - startTime
    total = f.total
    try:
        winSize = struct.unpack('4H', fcntl.ioctl(0, tty.TIOCGWINSZ, '12345679'))
    except OSError:
        winSize = [None, 80]
    if diff == 0.0:
        speed = 0.0
    else:
        speed = total / diff
    if speed:
        timeLeft = (f.size - total) / speed
    else:
        timeLeft = 0
    front = f.name
    if f.size:
        percentage = total / f.size * 100
    else:
        percentage = 100
    back = '%3i%% %s %sps %s ' % (percentage, self._abbrevSize(total), self._abbrevSize(speed), self._abbrevTime(timeLeft))
    spaces = (winSize[1] - (len(front) + len(back) + 1)) * ' '
    command = f'\r{front}{spaces}{back}'
    self._writeToTransport(command)