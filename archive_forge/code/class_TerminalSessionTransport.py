from typing import Dict
from zope.interface import implementer
from twisted.conch import avatar, error as econch, interfaces as iconch
from twisted.conch.insults import insults
from twisted.conch.ssh import factory, session
from twisted.python import components
class TerminalSessionTransport:

    def __init__(self, proto, chainedProtocol, avatar, width, height):
        self.proto = proto
        self.avatar = avatar
        self.chainedProtocol = chainedProtocol
        protoSession = self.proto.session
        self.proto.makeConnection(_Glue(write=self.chainedProtocol.dataReceived, loseConnection=lambda: avatar.conn.sendClose(protoSession), name='SSH Proto Transport'))

        def loseConnection():
            self.proto.loseConnection()
        self.chainedProtocol.makeConnection(_Glue(write=self.proto.write, loseConnection=loseConnection, name='Chained Proto Transport'))
        self.chainedProtocol.terminalProtocol.terminalSize(width, height)