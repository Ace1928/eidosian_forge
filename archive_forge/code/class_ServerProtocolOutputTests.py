import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
class ServerProtocolOutputTests(unittest.TestCase):
    """
    Tests for the bytes L{ServerProtocol} writes to its transport when its
    methods are called.
    """
    ESC = _ecmaCodeTableCoordinate(1, 11)
    CSI = ESC + _ecmaCodeTableCoordinate(5, 11)

    def setUp(self):
        self.protocol = ServerProtocol()
        self.transport = StringTransport()
        self.protocol.makeConnection(self.transport)

    def test_cursorUp(self):
        """
        L{ServerProtocol.cursorUp} writes the control sequence
        ending with L{CSFinalByte.CUU} to its transport.
        """
        self.protocol.cursorUp(1)
        self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.CUU.value)

    def test_cursorDown(self):
        """
        L{ServerProtocol.cursorDown} writes the control sequence
        ending with L{CSFinalByte.CUD} to its transport.
        """
        self.protocol.cursorDown(1)
        self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.CUD.value)

    def test_cursorForward(self):
        """
        L{ServerProtocol.cursorForward} writes the control sequence
        ending with L{CSFinalByte.CUF} to its transport.
        """
        self.protocol.cursorForward(1)
        self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.CUF.value)

    def test_cursorBackward(self):
        """
        L{ServerProtocol.cursorBackward} writes the control sequence
        ending with L{CSFinalByte.CUB} to its transport.
        """
        self.protocol.cursorBackward(1)
        self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.CUB.value)

    def test_cursorPosition(self):
        """
        L{ServerProtocol.cursorPosition} writes a control sequence
        ending with L{CSFinalByte.CUP} and containing the expected
        coordinates to its transport.
        """
        self.protocol.cursorPosition(0, 0)
        self.assertEqual(self.transport.value(), self.CSI + b'1;1' + CSFinalByte.CUP.value)

    def test_cursorHome(self):
        """
        L{ServerProtocol.cursorHome} writes a control sequence ending
        with L{CSFinalByte.CUP} and no parameters, so that the client
        defaults to (1, 1).
        """
        self.protocol.cursorHome()
        self.assertEqual(self.transport.value(), self.CSI + CSFinalByte.CUP.value)

    def test_index(self):
        """
        L{ServerProtocol.index} writes the control sequence ending in
        the 8-bit code table coordinates 4, 4.

        Note that ECMA48 5th Edition removes C{IND}.
        """
        self.protocol.index()
        self.assertEqual(self.transport.value(), self.ESC + _ecmaCodeTableCoordinate(4, 4))

    def test_reverseIndex(self):
        """
        L{ServerProtocol.reverseIndex} writes the control sequence
        ending in the L{C1SevenBit.RI}.
        """
        self.protocol.reverseIndex()
        self.assertEqual(self.transport.value(), self.ESC + C1SevenBit.RI.value)

    def test_nextLine(self):
        """
        L{ServerProtocol.nextLine} writes C{"\r
"} to its transport.
        """
        self.protocol.nextLine()
        self.assertEqual(self.transport.value(), b'\r\n')

    def test_setModes(self):
        """
        L{ServerProtocol.setModes} writes a control sequence
        containing the requested modes and ending in the
        L{CSFinalByte.SM}.
        """
        modesToSet = [modes.KAM, modes.IRM, modes.LNM]
        self.protocol.setModes(modesToSet)
        self.assertEqual(self.transport.value(), self.CSI + b';'.join((b'%d' % (m,) for m in modesToSet)) + CSFinalByte.SM.value)

    def test_setPrivateModes(self):
        """
        L{ServerProtocol.setPrivatesModes} writes a control sequence
        containing the requested private modes and ending in the
        L{CSFinalByte.SM}.
        """
        privateModesToSet = [privateModes.ERROR, privateModes.COLUMN, privateModes.ORIGIN]
        self.protocol.setModes(privateModesToSet)
        self.assertEqual(self.transport.value(), self.CSI + b';'.join((b'%d' % (m,) for m in privateModesToSet)) + CSFinalByte.SM.value)

    def test_resetModes(self):
        """
        L{ServerProtocol.resetModes} writes the control sequence
        ending in the L{CSFinalByte.RM}.
        """
        modesToSet = [modes.KAM, modes.IRM, modes.LNM]
        self.protocol.resetModes(modesToSet)
        self.assertEqual(self.transport.value(), self.CSI + b';'.join((b'%d' % (m,) for m in modesToSet)) + CSFinalByte.RM.value)

    def test_singleShift2(self):
        """
        L{ServerProtocol.singleShift2} writes an escape sequence
        followed by L{C1SevenBit.SS2}
        """
        self.protocol.singleShift2()
        self.assertEqual(self.transport.value(), self.ESC + C1SevenBit.SS2.value)

    def test_singleShift3(self):
        """
        L{ServerProtocol.singleShift3} writes an escape sequence
        followed by L{C1SevenBit.SS3}
        """
        self.protocol.singleShift3()
        self.assertEqual(self.transport.value(), self.ESC + C1SevenBit.SS3.value)

    def test_selectGraphicRendition(self):
        """
        L{ServerProtocol.selectGraphicRendition} writes a control
        sequence containing the requested attributes and ending with
        L{CSFinalByte.SGR}
        """
        self.protocol.selectGraphicRendition(str(BLINK), str(UNDERLINE))
        self.assertEqual(self.transport.value(), self.CSI + b'%d;%d' % (BLINK, UNDERLINE) + CSFinalByte.SGR.value)

    def test_horizontalTabulationSet(self):
        """
        L{ServerProtocol.horizontalTabulationSet} writes the escape
        sequence ending in L{C1SevenBit.HTS}
        """
        self.protocol.horizontalTabulationSet()
        self.assertEqual(self.transport.value(), self.ESC + C1SevenBit.HTS.value)

    def test_eraseToLineEnd(self):
        """
        L{ServerProtocol.eraseToLineEnd} writes the control sequence
        sequence ending in L{CSFinalByte.EL} and no parameters,
        forcing the client to default to 0 (from the active present
        position's current location to the end of the line.)
        """
        self.protocol.eraseToLineEnd()
        self.assertEqual(self.transport.value(), self.CSI + CSFinalByte.EL.value)

    def test_eraseToLineBeginning(self):
        """
        L{ServerProtocol.eraseToLineBeginning} writes the control
        sequence sequence ending in L{CSFinalByte.EL} and a parameter
        of 1 (from the beginning of the line up to and include the
        active present position's current location.)
        """
        self.protocol.eraseToLineBeginning()
        self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.EL.value)

    def test_eraseLine(self):
        """
        L{ServerProtocol.eraseLine} writes the control
        sequence sequence ending in L{CSFinalByte.EL} and a parameter
        of 2 (the entire line.)
        """
        self.protocol.eraseLine()
        self.assertEqual(self.transport.value(), self.CSI + b'2' + CSFinalByte.EL.value)

    def test_eraseToDisplayEnd(self):
        """
        L{ServerProtocol.eraseToDisplayEnd} writes the control
        sequence sequence ending in L{CSFinalByte.ED} and no parameters,
        forcing the client to default to 0 (from the active present
        position's current location to the end of the page.)
        """
        self.protocol.eraseToDisplayEnd()
        self.assertEqual(self.transport.value(), self.CSI + CSFinalByte.ED.value)

    def test_eraseToDisplayBeginning(self):
        """
        L{ServerProtocol.eraseToDisplayBeginning} writes the control
        sequence sequence ending in L{CSFinalByte.ED} a parameter of 1
        (from the beginning of the page up to and include the active
        present position's current location.)
        """
        self.protocol.eraseToDisplayBeginning()
        self.assertEqual(self.transport.value(), self.CSI + b'1' + CSFinalByte.ED.value)

    def test_eraseToDisplay(self):
        """
        L{ServerProtocol.eraseDisplay} writes the control sequence
        sequence ending in L{CSFinalByte.ED} a parameter of 2 (the
        entire page)
        """
        self.protocol.eraseDisplay()
        self.assertEqual(self.transport.value(), self.CSI + b'2' + CSFinalByte.ED.value)

    def test_deleteCharacter(self):
        """
        L{ServerProtocol.deleteCharacter} writes the control sequence
        containing the number of characters to delete and ending in
        L{CSFinalByte.DCH}
        """
        self.protocol.deleteCharacter(4)
        self.assertEqual(self.transport.value(), self.CSI + b'4' + CSFinalByte.DCH.value)

    def test_insertLine(self):
        """
        L{ServerProtocol.insertLine} writes the control sequence
        containing the number of lines to insert and ending in
        L{CSFinalByte.IL}
        """
        self.protocol.insertLine(5)
        self.assertEqual(self.transport.value(), self.CSI + b'5' + CSFinalByte.IL.value)

    def test_deleteLine(self):
        """
        L{ServerProtocol.deleteLine} writes the control sequence
        containing the number of lines to delete and ending in
        L{CSFinalByte.DL}
        """
        self.protocol.deleteLine(6)
        self.assertEqual(self.transport.value(), self.CSI + b'6' + CSFinalByte.DL.value)

    def test_setScrollRegionNoArgs(self):
        """
        With no arguments, L{ServerProtocol.setScrollRegion} writes a
        control sequence with no parameters, but a parameter
        separator, and ending in C{b'r'}.
        """
        self.protocol.setScrollRegion()
        self.assertEqual(self.transport.value(), self.CSI + b';' + b'r')

    def test_setScrollRegionJustFirst(self):
        """
        With just a value for its C{first} argument,
        L{ServerProtocol.setScrollRegion} writes a control sequence with
        that parameter, a parameter separator, and finally a C{b'r'}.
        """
        self.protocol.setScrollRegion(first=1)
        self.assertEqual(self.transport.value(), self.CSI + b'1;' + b'r')

    def test_setScrollRegionJustLast(self):
        """
        With just a value for its C{last} argument,
        L{ServerProtocol.setScrollRegion} writes a control sequence with
        a parameter separator, that parameter, and finally a C{b'r'}.
        """
        self.protocol.setScrollRegion(last=1)
        self.assertEqual(self.transport.value(), self.CSI + b';1' + b'r')

    def test_setScrollRegionFirstAndLast(self):
        """
        When given both C{first} and C{last}
        L{ServerProtocol.setScrollRegion} writes a control sequence with
        the first parameter, a parameter separator, the last
        parameter, and finally a C{b'r'}.
        """
        self.protocol.setScrollRegion(first=1, last=2)
        self.assertEqual(self.transport.value(), self.CSI + b'1;2' + b'r')

    def test_reportCursorPosition(self):
        """
        L{ServerProtocol.reportCursorPosition} writes a control
        sequence ending in L{CSFinalByte.DSR} with a parameter of 6
        (the Device Status Report returns the current active
        position.)
        """
        self.protocol.reportCursorPosition()
        self.assertEqual(self.transport.value(), self.CSI + b'6' + CSFinalByte.DSR.value)