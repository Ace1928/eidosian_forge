import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
class ServerFunctionKeysTests(ByteGroupingsMixin, unittest.TestCase):
    """Test for parsing and dispatching function keys (F1 - F12)"""
    protocolFactory = ServerProtocol
    byteList = []
    for byteCodes in (b'OP', b'OQ', b'OR', b'OS', b'15~', b'17~', b'18~', b'19~', b'20~', b'21~', b'23~', b'24~'):
        byteList.append(b'\x1b[' + byteCodes)
    TEST_BYTES = b''.join(byteList)
    del byteList, byteCodes

    def verifyResults(self, transport, proto, parser):
        ByteGroupingsMixin.verifyResults(self, transport, proto, parser)
        for funcNum in range(1, 13):
            funcArg = getattr(parser, 'F%d' % (funcNum,))
            result = self.assertCall(occurrences(proto).pop(0), 'keystrokeReceived', (funcArg, None))
            self.assertEqual(occurrences(result), [])
        self.assertFalse(occurrences(proto))