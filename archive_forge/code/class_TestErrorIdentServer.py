import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
class TestErrorIdentServer(ident.IdentServer):

    def lookup(self, serverAddress, clientAddress):
        raise self.exceptionType()