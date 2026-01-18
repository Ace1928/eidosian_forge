import code
import sys
import tokenize
from io import BytesIO
from traceback import format_exception
from types import TracebackType
from typing import Type
from twisted.conch import recvline
from twisted.internet import defer
from twisted.python.compat import _get_async_param
from twisted.python.htmlizer import TokenPrinter
from twisted.python.monkey import MonkeyPatcher
class ColoredManhole(Manhole):
    """
    A REPL which syntax colors input as users type it.
    """

    def getSource(self):
        """
        Return a string containing the currently entered source.

        This is only the code which will be considered for execution
        next.
        """
        return b'\n'.join(self.interpreter.buffer) + b'\n' + b''.join(self.lineBuffer)

    def characterReceived(self, ch, moreCharactersComing):
        if self.mode == 'insert':
            self.lineBuffer.insert(self.lineBufferIndex, ch)
        else:
            self.lineBuffer[self.lineBufferIndex:self.lineBufferIndex + 1] = [ch]
        self.lineBufferIndex += 1
        if moreCharactersComing:
            return
        if ch == b' ':
            self.terminal.write(ch)
            return
        source = self.getSource()
        try:
            coloredLine = lastColorizedLine(source)
        except tokenize.TokenError:
            self.terminal.write(ch)
        else:
            self.terminal.eraseLine()
            self.terminal.cursorBackward(len(self.lineBuffer) + len(self.ps[self.pn]) - 1)
            self.terminal.write(self.ps[self.pn] + coloredLine)
            n = len(self.lineBuffer) - self.lineBufferIndex
            if n:
                self.terminal.cursorBackward(n)