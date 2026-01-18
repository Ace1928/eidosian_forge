import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
class ProcMixinTests(unittest.TestCase):
    line = '4: %s:0019 %s:02FA 0A 00000000:00000000 00:00000000 00000000     0        0 10927 1 f72a5b80 3000 0 0 2 -1' % (_addr1, _addr2)
    sampleFile = '  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n   ' + line

    def testDottedQuadFromHexString(self):
        p = ident.ProcServerMixin()
        self.assertEqual(p.dottedQuadFromHexString(_addr1), '127.0.0.1')

    def testUnpackAddress(self):
        p = ident.ProcServerMixin()
        self.assertEqual(p.unpackAddress(_addr1 + ':0277'), ('127.0.0.1', 631))

    def testLineParser(self):
        p = ident.ProcServerMixin()
        self.assertEqual(p.parseLine(self.line), (('127.0.0.1', 25), ('1.2.3.4', 762), 0))

    def testExistingAddress(self):
        username = []
        p = ident.ProcServerMixin()
        p.entries = lambda: iter([self.line])
        p.getUsername = lambda uid: (username.append(uid), 'root')[1]
        self.assertEqual(p.lookup(('127.0.0.1', 25), ('1.2.3.4', 762)), (p.SYSTEM_NAME, 'root'))
        self.assertEqual(username, [0])

    def testNonExistingAddress(self):
        p = ident.ProcServerMixin()
        p.entries = lambda: iter([self.line])
        self.assertRaises(ident.NoUser, p.lookup, ('127.0.0.1', 26), ('1.2.3.4', 762))
        self.assertRaises(ident.NoUser, p.lookup, ('127.0.0.1', 25), ('1.2.3.5', 762))
        self.assertRaises(ident.NoUser, p.lookup, ('127.0.0.1', 25), ('1.2.3.4', 763))

    def testLookupProcNetTcp(self):
        """
        L{ident.ProcServerMixin.lookup} uses the Linux TCP process table.
        """
        open_calls = []

        def mocked_open(*args, **kwargs):
            """
            Mock for the open call to prevent actually opening /proc/net/tcp.
            """
            open_calls.append((args, kwargs))
            return StringIO(self.sampleFile)
        self.patch(builtins, 'open', mocked_open)
        p = ident.ProcServerMixin()
        self.assertRaises(ident.NoUser, p.lookup, ('127.0.0.1', 26), ('1.2.3.4', 762))
        self.assertEqual([(('/proc/net/tcp',), {})], open_calls)