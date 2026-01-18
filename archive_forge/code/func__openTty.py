import contextlib
import getpass
import io
import os
import sys
from base64 import decodebytes
from twisted.conch.client import agent
from twisted.conch.client.knownhosts import ConsoleUI, KnownHostsFile
from twisted.conch.error import ConchError
from twisted.conch.ssh import common, keys, userauth
from twisted.internet import defer, protocol, reactor
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
@classmethod
def _openTty(cls):
    """
        Open /dev/tty as two streams one in read, one in write mode,
        and return them.

        @return: File objects for reading and writing to /dev/tty,
                 corresponding to standard input and standard output.
        @rtype: A L{tuple} of L{io.TextIOWrapper} on Python 3.
        """
    stdin = io.TextIOWrapper(open('/dev/tty', 'rb'))
    stdout = io.TextIOWrapper(open('/dev/tty', 'wb'))
    return (stdin, stdout)