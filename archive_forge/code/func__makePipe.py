import errno
import os
import sys
import warnings
from os import close, pathsep, pipe, read
from socket import AF_INET, AF_INET6, SOL_SOCKET, error, socket
from struct import pack
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.internet.error import ProcessDone
from twisted.internet.protocol import ProcessProtocol
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def _makePipe():
    """
    Create a pipe, and return the two FDs wrapped in L{_FDHolders}.
    """
    r, w = pipe()
    return (_FDHolder(r), _FDHolder(w))