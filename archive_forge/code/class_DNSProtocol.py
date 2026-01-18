from __future__ import annotations
import inspect
import random
import socket
import struct
from io import BytesIO
from itertools import chain
from typing import Optional, Sequence, SupportsInt, Union, overload
from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import CannotListenError
from twisted.python import failure, log, randbytes, util as tputil
from twisted.python.compat import cmp, comparable, nativeString
from twisted.names.error import (
class DNSProtocol(DNSMixin, protocol.Protocol):
    """
    DNS protocol over TCP.
    """
    length = None
    buffer = b''

    def writeMessage(self, message):
        """
        Send a message holding DNS queries.

        @type message: L{Message}
        """
        s = message.toStr()
        self.transport.write(struct.pack('!H', len(s)) + s)

    def connectionMade(self):
        """
        Connection is made: reset internal state, and notify the controller.
        """
        self.liveMessages = {}
        self.controller.connectionMade(self)

    def connectionLost(self, reason):
        """
        Notify the controller that this protocol is no longer
        connected.
        """
        self.controller.connectionLost(self)

    def dataReceived(self, data):
        self.buffer += data
        while self.buffer:
            if self.length is None and len(self.buffer) >= 2:
                self.length = struct.unpack('!H', self.buffer[:2])[0]
                self.buffer = self.buffer[2:]
            if len(self.buffer) >= self.length:
                myChunk = self.buffer[:self.length]
                m = Message()
                m.fromStr(myChunk)
                try:
                    d, canceller = self.liveMessages[m.id]
                except KeyError:
                    self.controller.messageReceived(m, self)
                else:
                    del self.liveMessages[m.id]
                    canceller.cancel()
                    try:
                        d.callback(m)
                    except BaseException:
                        log.err()
                self.buffer = self.buffer[self.length:]
                self.length = None
            else:
                break

    def query(self, queries, timeout=60):
        """
        Send out a message with the given queries.

        @type queries: L{list} of C{Query} instances
        @param queries: The queries to transmit

        @rtype: C{Deferred}
        """
        id = self.pickID()
        return self._query(queries, timeout, id, self.writeMessage)