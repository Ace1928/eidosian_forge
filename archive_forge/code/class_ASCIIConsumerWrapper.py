import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
class ASCIIConsumerWrapper:

    def __init__(self, cons):
        self.cons = cons
        self.registerProducer = cons.registerProducer
        self.unregisterProducer = cons.unregisterProducer
        assert os.linesep == '\r\n' or len(os.linesep) == 1, 'Unsupported platform (yea right like this even exists)'
        if os.linesep == '\r\n':
            self.write = cons.write

    def write(self, bytes):
        return self.cons.write(bytes.replace(os.linesep, '\r\n'))