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
class _FDHolder:
    """
    A wrapper around a FD that will remember if it has been closed or not.
    """

    def __init__(self, fd):
        self._fd = fd

    def fileno(self):
        """
        Return the fileno of this FD.
        """
        return self._fd

    def close(self):
        """
        Close the FD. If it's already been closed, do nothing.
        """
        if self._fd:
            close(self._fd)
            self._fd = None

    def __del__(self):
        """
        If C{self._fd} is unclosed, raise a warning.
        """
        if self._fd:
            warnings.warn(f'FD {self._fd} was not closed!', ResourceWarning)
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()