import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
class SillyDescriptor(abstract.FileDescriptor):
    """
    A descriptor whose data buffer gets filled very fast.

    Useful for testing FileDescriptor's IConsumer interface, since
    the data buffer fills as soon as at least four characters are
    written to it, and gets emptied in a single doWrite() cycle.
    """
    bufferSize = 3
    connected = True

    def writeSomeData(self, data):
        """
        Always write all data.
        """
        return len(data)

    def startWriting(self):
        """
        Do nothing: bypass the reactor.
        """
    stopWriting = startWriting