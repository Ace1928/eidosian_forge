import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
class ClientControlSequencesTests(unittest.TestCase, MockMixin):

    def setUp(self):
        self.transport = StringTransport()
        self.proto = Mock()
        self.parser = ClientProtocol(lambda: self.proto)
        self.parser.factory = self
        self.parser.makeConnection(self.transport)
        result = self.assertCall(occurrences(self.proto).pop(0), 'makeConnection', (self.parser,))
        self.assertFalse(occurrences(result))

    def testSimpleCardinals(self):
        self.parser.dataReceived(b''.join((b'\x1b[' + n + ch for ch in iterbytes(b'BACD') for n in (b'', b'2', b'20', b'200'))))
        occs = occurrences(self.proto)
        for meth in ('Down', 'Up', 'Forward', 'Backward'):
            for count in (1, 2, 20, 200):
                result = self.assertCall(occs.pop(0), 'cursor' + meth, (count,))
                self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testScrollRegion(self):
        self.parser.dataReceived(b'\x1b[5;22r\x1b[r')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'setScrollRegion', (5, 22))
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'setScrollRegion', (None, None))
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testHeightAndWidth(self):
        self.parser.dataReceived(b'\x1b#3\x1b#4\x1b#5\x1b#6')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'doubleHeightLine', (True,))
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'doubleHeightLine', (False,))
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'singleWidthLine')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'doubleWidthLine')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testCharacterSet(self):
        self.parser.dataReceived(b''.join([b''.join([b'\x1b' + g + n for n in iterbytes(b'AB012')]) for g in iterbytes(b'()')]))
        occs = occurrences(self.proto)
        for which in (G0, G1):
            for charset in (CS_UK, CS_US, CS_DRAWING, CS_ALTERNATE, CS_ALTERNATE_SPECIAL):
                result = self.assertCall(occs.pop(0), 'selectCharacterSet', (charset, which))
                self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testShifting(self):
        self.parser.dataReceived(b'\x15\x14')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'shiftIn')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'shiftOut')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testSingleShifts(self):
        self.parser.dataReceived(b'\x1bN\x1bO')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'singleShift2')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'singleShift3')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testKeypadMode(self):
        self.parser.dataReceived(b'\x1b=\x1b>')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'applicationKeypadMode')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'numericKeypadMode')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testCursor(self):
        self.parser.dataReceived(b'\x1b7\x1b8')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'saveCursor')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'restoreCursor')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testReset(self):
        self.parser.dataReceived(b'\x1bc')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'reset')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testIndex(self):
        self.parser.dataReceived(b'\x1bD\x1bM\x1bE')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'index')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'reverseIndex')
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'nextLine')
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testModes(self):
        self.parser.dataReceived(b'\x1b[' + b';'.join((b'%d' % (m,) for m in [modes.KAM, modes.IRM, modes.LNM])) + b'h')
        self.parser.dataReceived(b'\x1b[' + b';'.join((b'%d' % (m,) for m in [modes.KAM, modes.IRM, modes.LNM])) + b'l')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'setModes', ([modes.KAM, modes.IRM, modes.LNM],))
        self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'resetModes', ([modes.KAM, modes.IRM, modes.LNM],))
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testErasure(self):
        self.parser.dataReceived(b'\x1b[K\x1b[1K\x1b[2K\x1b[J\x1b[1J\x1b[2J\x1b[3P')
        occs = occurrences(self.proto)
        for meth in ('eraseToLineEnd', 'eraseToLineBeginning', 'eraseLine', 'eraseToDisplayEnd', 'eraseToDisplayBeginning', 'eraseDisplay'):
            result = self.assertCall(occs.pop(0), meth)
            self.assertFalse(occurrences(result))
        result = self.assertCall(occs.pop(0), 'deleteCharacter', (3,))
        self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testLineDeletion(self):
        self.parser.dataReceived(b'\x1b[M\x1b[3M')
        occs = occurrences(self.proto)
        for arg in (1, 3):
            result = self.assertCall(occs.pop(0), 'deleteLine', (arg,))
            self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testLineInsertion(self):
        self.parser.dataReceived(b'\x1b[L\x1b[3L')
        occs = occurrences(self.proto)
        for arg in (1, 3):
            result = self.assertCall(occs.pop(0), 'insertLine', (arg,))
            self.assertFalse(occurrences(result))
        self.assertFalse(occs)

    def testCursorPosition(self):
        methods(self.proto)['reportCursorPosition'] = (6, 7)
        self.parser.dataReceived(b'\x1b[6n')
        self.assertEqual(self.transport.value(), b'\x1b[7;8R')
        occs = occurrences(self.proto)
        result = self.assertCall(occs.pop(0), 'reportCursorPosition')
        self.assertEqual(result, (6, 7))

    def test_applicationDataBytes(self):
        """
        Contiguous non-control bytes are passed to a single call to the
        C{write} method of the terminal to which the L{ClientProtocol} is
        connected.
        """
        occs = occurrences(self.proto)
        self.parser.dataReceived(b'a')
        self.assertCall(occs.pop(0), 'write', (b'a',))
        self.parser.dataReceived(b'bc')
        self.assertCall(occs.pop(0), 'write', (b'bc',))

    def _applicationDataTest(self, data, calls):
        occs = occurrences(self.proto)
        self.parser.dataReceived(data)
        while calls:
            self.assertCall(occs.pop(0), *calls.pop(0))
        self.assertFalse(occs, f'No other calls should happen: {occs!r}')

    def test_shiftInAfterApplicationData(self):
        """
        Application data bytes followed by a shift-in command are passed to a
        call to C{write} before the terminal's C{shiftIn} method is called.
        """
        self._applicationDataTest(b'ab\x15', [('write', (b'ab',)), ('shiftIn',)])

    def test_shiftOutAfterApplicationData(self):
        """
        Application data bytes followed by a shift-out command are passed to a
        call to C{write} before the terminal's C{shiftOut} method is called.
        """
        self._applicationDataTest(b'ab\x14', [('write', (b'ab',)), ('shiftOut',)])

    def test_cursorBackwardAfterApplicationData(self):
        """
        Application data bytes followed by a cursor-backward command are passed
        to a call to C{write} before the terminal's C{cursorBackward} method is
        called.
        """
        self._applicationDataTest(b'ab\x08', [('write', (b'ab',)), ('cursorBackward',)])

    def test_escapeAfterApplicationData(self):
        """
        Application data bytes followed by an escape character are passed to a
        call to C{write} before the terminal's handler method for the escape is
        called.
        """
        self._applicationDataTest(b'ab\x1bD', [('write', (b'ab',)), ('index',)])
        self._applicationDataTest(b'ab\x1b[4h', [('write', (b'ab',)), ('setModes', ([4],))])