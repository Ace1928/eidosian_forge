import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
class PrintableCharactersTests(ByteGroupingsMixin, unittest.TestCase):
    protocolFactory = ServerProtocol
    TEST_BYTES = b'abc123ABC!@#\x1ba\x1bb\x1bc\x1b1\x1b2\x1b3'

    def verifyResults(self, transport, proto, parser):
        ByteGroupingsMixin.verifyResults(self, transport, proto, parser)
        for char in iterbytes(b'abc123ABC!@#'):
            result = self.assertCall(occurrences(proto).pop(0), 'keystrokeReceived', (char, None))
            self.assertEqual(occurrences(result), [])
        for char in iterbytes(b'abc123'):
            result = self.assertCall(occurrences(proto).pop(0), 'keystrokeReceived', (char, parser.ALT))
            self.assertEqual(occurrences(result), [])
        occs = occurrences(proto)
        self.assertFalse(occs, f'{occs!r} should have been []')