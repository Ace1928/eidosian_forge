import getpass
import locale
import operator
import os
import struct
import sys
import time
from io import BytesIO, TextIOWrapper
from unittest import skipIf
from zope.interface import implementer
from twisted.conch import ls
from twisted.conch.interfaces import ISFTPFile
from twisted.conch.test.test_filetransfer import FileTransferTestAvatar, SFTPTestBase
from twisted.cred import portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.internet.utils import getProcessOutputAndValue, getProcessValue
from twisted.python import log
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def _getBatchOutput(self, f):
    fn = self.mktemp()
    with open(fn, 'w') as fp:
        fp.write(f)
    port = self.server.getHost().port
    cmds = '-p %i -l testuser --known-hosts kh_test --user-authentications publickey --host-key-algorithms ssh-rsa -i dsa_test -a -v -b %s 127.0.0.1' % (port, fn)
    cmds = test_conch._makeArgs(cmds.split(), mod='cftp')[1:]
    log.msg(f'running {sys.executable} {cmds}')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(sys.path)
    self.server.factory.expectedLoseConnection = 1
    d = getProcessOutputAndValue(sys.executable, cmds, env=env)

    def _cleanup(res):
        os.remove(fn)
        return res
    d.addCallback(lambda res: res[0])
    d.addBoth(_cleanup)
    return d