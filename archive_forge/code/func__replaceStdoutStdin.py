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
@contextlib.contextmanager
def _replaceStdoutStdin(cls):
    """
        Contextmanager that replaces stdout and stdin with /dev/tty
        and resets them when it is done.
        """
    oldout, oldin = (sys.stdout, sys.stdin)
    sys.stdin, sys.stdout = cls._openTty()
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdin.close()
        sys.stdout, sys.stdin = (oldout, oldin)