import os
import socket
import sys
from unittest import skipIf
from twisted.internet import address, defer, error, interfaces, protocol, reactor, utils
from twisted.python import lockfile
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.test.test_tcp import MyClientFactory, MyServerFactory
from twisted.trial import unittest
def _uncleanSocketTest(self, callback):
    self.filename = self.mktemp()
    source = networkString('from twisted.internet import protocol, reactor\nreactor.listenUNIX(%r, protocol.ServerFactory(),wantPID=True)\n' % (self.filename,))
    env = {b'PYTHONPATH': FilePath(os.pathsep.join(sys.path)).asBytesMode().path}
    pyExe = FilePath(sys.executable).asBytesMode().path
    d = utils.getProcessValue(pyExe, (b'-u', b'-c', source), env=env)
    d.addCallback(callback)
    return d