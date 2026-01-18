import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
class LocalTerminalBufferMixin:
    """
    A mixin for RecvLine subclasses which records the state of the terminal.

    This is accomplished by performing all L{ITerminalTransport} operations on both
    the transport passed to makeConnection and an instance of helper.TerminalBuffer.

    @ivar terminalCopy: A L{helper.TerminalBuffer} instance which efforts
    will be made to keep up to date with the actual terminal
    associated with this protocol instance.
    """

    def makeConnection(self, transport):
        self.terminalCopy = helper.TerminalBuffer()
        self.terminalCopy.connectionMade()
        return super().makeConnection(TransportSequence(transport, self.terminalCopy))

    def __str__(self) -> str:
        return str(self.terminalCopy)