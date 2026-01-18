import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
@implementer(insults.ITerminalTransport)
class TransportSequence:
    """
    An L{ITerminalTransport} implementation which forwards calls to
    one or more other L{ITerminalTransport}s.

    This is a cheap way for servers to keep track of the state they
    expect the client to see, since all terminal manipulations can be
    send to the real client and to a terminal emulator that lives in
    the server process.
    """
    for keyID in (b'UP_ARROW', b'DOWN_ARROW', b'RIGHT_ARROW', b'LEFT_ARROW', b'HOME', b'INSERT', b'DELETE', b'END', b'PGUP', b'PGDN', b'F1', b'F2', b'F3', b'F4', b'F5', b'F6', b'F7', b'F8', b'F9', b'F10', b'F11', b'F12'):
        execBytes = keyID + b' = object()'
        execStr = execBytes.decode('ascii')
        exec(execStr)
    TAB = b'\t'
    BACKSPACE = b'\x7f'

    def __init__(self, *transports):
        assert transports, 'Cannot construct a TransportSequence with no transports'
        self.transports = transports
    for method in insults.ITerminalTransport:
        exec('def %s(self, *a, **kw):\n    for tpt in self.transports:\n        result = tpt.%s(*a, **kw)\n    return result\n' % (method, method))

    def getHost(self):
        raise NotImplementedError('Unimplemented: TransportSequence.getHost')

    def getPeer(self):
        raise NotImplementedError('Unimplemented: TransportSequence.getPeer')

    def loseConnection(self):
        raise NotImplementedError('Unimplemented: TransportSequence.loseConnection')

    def write(self, data):
        raise NotImplementedError('Unimplemented: TransportSequence.write')

    def writeSequence(self, data):
        raise NotImplementedError('Unimplemented: TransportSequence.writeSequence')

    def cursorUp(self, n=1):
        raise NotImplementedError('Unimplemented: TransportSequence.cursorUp')

    def cursorDown(self, n=1):
        raise NotImplementedError('Unimplemented: TransportSequence.cursorDown')

    def cursorForward(self, n=1):
        raise NotImplementedError('Unimplemented: TransportSequence.cursorForward')

    def cursorBackward(self, n=1):
        raise NotImplementedError('Unimplemented: TransportSequence.cursorBackward')

    def cursorPosition(self, column, line):
        raise NotImplementedError('Unimplemented: TransportSequence.cursorPosition')

    def cursorHome(self):
        raise NotImplementedError('Unimplemented: TransportSequence.cursorHome')

    def index(self):
        raise NotImplementedError('Unimplemented: TransportSequence.index')

    def reverseIndex(self):
        raise NotImplementedError('Unimplemented: TransportSequence.reverseIndex')

    def nextLine(self):
        raise NotImplementedError('Unimplemented: TransportSequence.nextLine')

    def saveCursor(self):
        raise NotImplementedError('Unimplemented: TransportSequence.saveCursor')

    def restoreCursor(self):
        raise NotImplementedError('Unimplemented: TransportSequence.restoreCursor')

    def setModes(self, modes):
        raise NotImplementedError('Unimplemented: TransportSequence.setModes')

    def resetModes(self, mode):
        raise NotImplementedError('Unimplemented: TransportSequence.resetModes')

    def setPrivateModes(self, modes):
        raise NotImplementedError('Unimplemented: TransportSequence.setPrivateModes')

    def resetPrivateModes(self, modes):
        raise NotImplementedError('Unimplemented: TransportSequence.resetPrivateModes')

    def applicationKeypadMode(self):
        raise NotImplementedError('Unimplemented: TransportSequence.applicationKeypadMode')

    def numericKeypadMode(self):
        raise NotImplementedError('Unimplemented: TransportSequence.numericKeypadMode')

    def selectCharacterSet(self, charSet, which):
        raise NotImplementedError('Unimplemented: TransportSequence.selectCharacterSet')

    def shiftIn(self):
        raise NotImplementedError('Unimplemented: TransportSequence.shiftIn')

    def shiftOut(self):
        raise NotImplementedError('Unimplemented: TransportSequence.shiftOut')

    def singleShift2(self):
        raise NotImplementedError('Unimplemented: TransportSequence.singleShift2')

    def singleShift3(self):
        raise NotImplementedError('Unimplemented: TransportSequence.singleShift3')

    def selectGraphicRendition(self, *attributes):
        raise NotImplementedError('Unimplemented: TransportSequence.selectGraphicRendition')

    def horizontalTabulationSet(self):
        raise NotImplementedError('Unimplemented: TransportSequence.horizontalTabulationSet')

    def tabulationClear(self):
        raise NotImplementedError('Unimplemented: TransportSequence.tabulationClear')

    def tabulationClearAll(self):
        raise NotImplementedError('Unimplemented: TransportSequence.tabulationClearAll')

    def doubleHeightLine(self, top=True):
        raise NotImplementedError('Unimplemented: TransportSequence.doubleHeightLine')

    def singleWidthLine(self):
        raise NotImplementedError('Unimplemented: TransportSequence.singleWidthLine')

    def doubleWidthLine(self):
        raise NotImplementedError('Unimplemented: TransportSequence.doubleWidthLine')

    def eraseToLineEnd(self):
        raise NotImplementedError('Unimplemented: TransportSequence.eraseToLineEnd')

    def eraseToLineBeginning(self):
        raise NotImplementedError('Unimplemented: TransportSequence.eraseToLineBeginning')

    def eraseLine(self):
        raise NotImplementedError('Unimplemented: TransportSequence.eraseLine')

    def eraseToDisplayEnd(self):
        raise NotImplementedError('Unimplemented: TransportSequence.eraseToDisplayEnd')

    def eraseToDisplayBeginning(self):
        raise NotImplementedError('Unimplemented: TransportSequence.eraseToDisplayBeginning')

    def eraseDisplay(self):
        raise NotImplementedError('Unimplemented: TransportSequence.eraseDisplay')

    def deleteCharacter(self, n=1):
        raise NotImplementedError('Unimplemented: TransportSequence.deleteCharacter')

    def insertLine(self, n=1):
        raise NotImplementedError('Unimplemented: TransportSequence.insertLine')

    def deleteLine(self, n=1):
        raise NotImplementedError('Unimplemented: TransportSequence.deleteLine')

    def reportCursorPosition(self):
        raise NotImplementedError('Unimplemented: TransportSequence.reportCursorPosition')

    def reset(self):
        raise NotImplementedError('Unimplemented: TransportSequence.reset')

    def unhandledControlSequence(self, seq):
        raise NotImplementedError('Unimplemented: TransportSequence.unhandledControlSequence')