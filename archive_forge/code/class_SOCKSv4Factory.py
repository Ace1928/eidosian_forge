import socket
import string
import struct
import time
from twisted.internet import defer, protocol, reactor
from twisted.python import log
class SOCKSv4Factory(protocol.Factory):
    """
    A factory for a SOCKSv4 proxy.

    Constructor accepts one argument, a log file name.
    """

    def __init__(self, log):
        self.logging = log

    def buildProtocol(self, addr):
        return SOCKSv4(self.logging, reactor)