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
class DNSDatagramProtocol(DNSMixin, protocol.DatagramProtocol):
    """
    DNS protocol over UDP.
    """
    resends = None

    def stopProtocol(self):
        """
        Stop protocol: reset state variables.
        """
        self.liveMessages = {}
        self.resends = {}
        self.transport = None

    def startProtocol(self):
        """
        Upon start, reset internal state.
        """
        self.liveMessages = {}
        self.resends = {}

    def writeMessage(self, message, address):
        """
        Send a message holding DNS queries.

        @type message: L{Message}
        """
        self.transport.write(message.toStr(), address)

    def startListening(self):
        self._reactor.listenUDP(0, self, maxPacketSize=512)

    def datagramReceived(self, data, addr):
        """
        Read a datagram, extract the message in it and trigger the associated
        Deferred.
        """
        m = Message()
        try:
            m.fromStr(data)
        except EOFError:
            log.msg('Truncated packet (%d bytes) from %s' % (len(data), addr))
            return
        except ValueError as ex:
            log.msg(f'Invalid packet ({ex}) from {addr}')
            return
        except BaseException:
            log.err(failure.Failure(), 'Unexpected decoding error')
            return
        if m.id in self.liveMessages:
            d, canceller = self.liveMessages[m.id]
            del self.liveMessages[m.id]
            canceller.cancel()
            try:
                d.callback(m)
            except BaseException:
                log.err()
        elif m.id not in self.resends:
            self.controller.messageReceived(m, self, addr)

    def removeResend(self, id):
        """
        Mark message ID as no longer having duplication suppression.
        """
        try:
            del self.resends[id]
        except KeyError:
            pass

    def query(self, address, queries, timeout=10, id=None):
        """
        Send out a message with the given queries.

        @type address: L{tuple} of L{str} and L{int}
        @param address: The address to which to send the query

        @type queries: L{list} of C{Query} instances
        @param queries: The queries to transmit

        @rtype: C{Deferred}
        """
        if not self.transport:
            try:
                self.startListening()
            except CannotListenError:
                return defer.fail()
        if id is None:
            id = self.pickID()
        else:
            self.resends[id] = 1

        def writeMessage(m):
            self.writeMessage(m, address)
        return self._query(queries, timeout, id, writeMessage)