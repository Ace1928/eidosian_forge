import os
import sys
from unittest import skipIf
from twisted.conch import recvline
from twisted.conch.insults import insults
from twisted.cred import portal
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.python import components, filepath, reflect
from twisted.python.compat import iterbytes
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
from twisted.conch import telnet
from twisted.conch.insults import helper
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import checkers
from twisted.conch.test import test_telnet
class ArrowsTests(TestCase):

    def setUp(self):
        self.underlyingTransport = StringTransport()
        self.pt = insults.ServerProtocol()
        self.p = recvline.HistoricRecvLine()
        self.pt.protocolFactory = lambda: self.p
        self.pt.factory = self
        self.pt.makeConnection(self.underlyingTransport)

    def test_printableCharacters(self):
        """
        When L{HistoricRecvLine} receives a printable character,
        it adds it to the current line buffer.
        """
        self.p.keystrokeReceived(b'x', None)
        self.p.keystrokeReceived(b'y', None)
        self.p.keystrokeReceived(b'z', None)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))

    def test_horizontalArrows(self):
        """
        When L{HistoricRecvLine} receives a LEFT_ARROW or
        RIGHT_ARROW keystroke it moves the cursor left or right
        in the current line buffer, respectively.
        """
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz'):
            kR(ch)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.RIGHT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.LEFT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'xy', b'z'))
        kR(self.pt.LEFT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'x', b'yz'))
        kR(self.pt.LEFT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b'xyz'))
        kR(self.pt.LEFT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b'xyz'))
        kR(self.pt.RIGHT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'x', b'yz'))
        kR(self.pt.RIGHT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'xy', b'z'))
        kR(self.pt.RIGHT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.RIGHT_ARROW)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))

    def test_newline(self):
        """
        When {HistoricRecvLine} receives a newline, it adds the current
        line buffer to the end of its history buffer.
        """
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz\nabc\n123\n'):
            kR(ch)
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc', b'123'), ()))
        kR(b'c')
        kR(b'b')
        kR(b'a')
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc', b'123'), ()))
        kR(b'\n')
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc', b'123', b'cba'), ()))

    def test_verticalArrows(self):
        """
        When L{HistoricRecvLine} receives UP_ARROW or DOWN_ARROW
        keystrokes it move the current index in the current history
        buffer up or down, and resets the current line buffer to the
        previous or next line in history, respectively for each.
        """
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz\nabc\n123\n'):
            kR(ch)
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc', b'123'), ()))
        self.assertEqual(self.p.currentLineBuffer(), (b'', b''))
        kR(self.pt.UP_ARROW)
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc'), (b'123',)))
        self.assertEqual(self.p.currentLineBuffer(), (b'123', b''))
        kR(self.pt.UP_ARROW)
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz',), (b'abc', b'123')))
        self.assertEqual(self.p.currentLineBuffer(), (b'abc', b''))
        kR(self.pt.UP_ARROW)
        self.assertEqual(self.p.currentHistoryBuffer(), ((), (b'xyz', b'abc', b'123')))
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.UP_ARROW)
        self.assertEqual(self.p.currentHistoryBuffer(), ((), (b'xyz', b'abc', b'123')))
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        for i in range(4):
            kR(self.pt.DOWN_ARROW)
        self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc', b'123'), ()))

    def test_home(self):
        """
        When L{HistoricRecvLine} receives a HOME keystroke it moves the
        cursor to the beginning of the current line buffer.
        """
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'hello, world'):
            kR(ch)
        self.assertEqual(self.p.currentLineBuffer(), (b'hello, world', b''))
        kR(self.pt.HOME)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b'hello, world'))

    def test_end(self):
        """
        When L{HistoricRecvLine} receives an END keystroke it moves the cursor
        to the end of the current line buffer.
        """
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'hello, world'):
            kR(ch)
        self.assertEqual(self.p.currentLineBuffer(), (b'hello, world', b''))
        kR(self.pt.HOME)
        kR(self.pt.END)
        self.assertEqual(self.p.currentLineBuffer(), (b'hello, world', b''))

    def test_backspace(self):
        """
        When L{HistoricRecvLine} receives a BACKSPACE keystroke it deletes
        the character immediately before the cursor.
        """
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz'):
            kR(ch)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.BACKSPACE)
        self.assertEqual(self.p.currentLineBuffer(), (b'xy', b''))
        kR(self.pt.LEFT_ARROW)
        kR(self.pt.BACKSPACE)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b'y'))
        kR(self.pt.BACKSPACE)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b'y'))

    def test_delete(self):
        """
        When L{HistoricRecvLine} receives a DELETE keystroke, it
        delets the character immediately after the cursor.
        """
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz'):
            kR(ch)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.DELETE)
        self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
        kR(self.pt.LEFT_ARROW)
        kR(self.pt.DELETE)
        self.assertEqual(self.p.currentLineBuffer(), (b'xy', b''))
        kR(self.pt.LEFT_ARROW)
        kR(self.pt.DELETE)
        self.assertEqual(self.p.currentLineBuffer(), (b'x', b''))
        kR(self.pt.LEFT_ARROW)
        kR(self.pt.DELETE)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b''))
        kR(self.pt.DELETE)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b''))

    def test_insert(self):
        """
        When not in INSERT mode, L{HistoricRecvLine} inserts the typed
        character at the cursor before the next character.
        """
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz'):
            kR(ch)
        kR(self.pt.LEFT_ARROW)
        kR(b'A')
        self.assertEqual(self.p.currentLineBuffer(), (b'xyA', b'z'))
        kR(self.pt.LEFT_ARROW)
        kR(b'B')
        self.assertEqual(self.p.currentLineBuffer(), (b'xyB', b'Az'))

    def test_typeover(self):
        """
        When in INSERT mode and upon receiving a keystroke with a printable
        character, L{HistoricRecvLine} replaces the character at
        the cursor with the typed character rather than inserting before.
        Ah, the ironies of INSERT mode.
        """
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        for ch in iterbytes(b'xyz'):
            kR(ch)
        kR(self.pt.INSERT)
        kR(self.pt.LEFT_ARROW)
        kR(b'A')
        self.assertEqual(self.p.currentLineBuffer(), (b'xyA', b''))
        kR(self.pt.LEFT_ARROW)
        kR(b'B')
        self.assertEqual(self.p.currentLineBuffer(), (b'xyB', b''))

    def test_unprintableCharacters(self):
        """
        When L{HistoricRecvLine} receives a keystroke for an unprintable
        function key with no assigned behavior, the line buffer is unmodified.
        """
        kR = lambda ch: self.p.keystrokeReceived(ch, None)
        pt = self.pt
        for ch in (pt.F1, pt.F2, pt.F3, pt.F4, pt.F5, pt.F6, pt.F7, pt.F8, pt.F9, pt.F10, pt.F11, pt.F12, pt.PGUP, pt.PGDN):
            kR(ch)
            self.assertEqual(self.p.currentLineBuffer(), (b'', b''))