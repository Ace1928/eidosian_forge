import socket
import time
import warnings
from collections import OrderedDict
from typing import Dict, List
from zope.interface import Interface, implementer
from twisted import cred
from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic
from twisted.python import log
class MessagesParser(basic.LineReceiver):
    """
    A SIP messages parser.

    Expects dataReceived, dataDone repeatedly,
    in that order. Shouldn't be connected to actual transport.
    """
    version = 'SIP/2.0'
    acceptResponses = 1
    acceptRequests = 1
    state = 'firstline'
    debug = 0

    def __init__(self, messageReceivedCallback):
        self.messageReceived = messageReceivedCallback
        self.reset()

    def reset(self, remainingData=''):
        self.state = 'firstline'
        self.length = None
        self.bodyReceived = 0
        self.message = None
        self.header = None
        self.setLineMode(remainingData)

    def invalidMessage(self):
        self.state = 'invalid'
        self.setRawMode()

    def dataDone(self):
        """
        Clear out any buffered data that may be hanging around.
        """
        self.clearLineBuffer()
        if self.state == 'firstline':
            return
        if self.state != 'body':
            self.reset()
            return
        if self.length == None:
            self.messageDone()
        elif self.length < self.bodyReceived:
            self.reset()
        else:
            raise RuntimeError('this should never happen')

    def dataReceived(self, data):
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            basic.LineReceiver.dataReceived(self, data)
        except Exception:
            log.err()
            self.invalidMessage()

    def handleFirstLine(self, line):
        """
        Expected to create self.message.
        """
        raise NotImplementedError

    def lineLengthExceeded(self, line):
        self.invalidMessage()

    def lineReceived(self, line):
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        if self.state == 'firstline':
            while line.startswith('\n') or line.startswith('\r'):
                line = line[1:]
            if not line:
                return
            try:
                a, b, c = line.split(' ', 2)
            except ValueError:
                self.invalidMessage()
                return
            if a == 'SIP/2.0' and self.acceptResponses:
                try:
                    code = int(b)
                except ValueError:
                    self.invalidMessage()
                    return
                self.message = Response(code, c)
            elif c == 'SIP/2.0' and self.acceptRequests:
                self.message = Request(a, b)
            else:
                self.invalidMessage()
                return
            self.state = 'headers'
            return
        else:
            assert self.state == 'headers'
        if line:
            if line.startswith(' ') or line.startswith('\t'):
                name, value = self.header
                self.header = (name, value + line.lstrip())
            else:
                if self.header:
                    self.message.addHeader(*self.header)
                    self.header = None
                try:
                    name, value = line.split(':', 1)
                except ValueError:
                    self.invalidMessage()
                    return
                self.header = (name, value.lstrip())
                if name.lower() == 'content-length':
                    try:
                        self.length = int(value.lstrip())
                    except ValueError:
                        self.invalidMessage()
                        return
        else:
            self.state = 'body'
            if self.header:
                self.message.addHeader(*self.header)
                self.header = None
            if self.length == 0:
                self.messageDone()
                return
            self.setRawMode()

    def messageDone(self, remainingData=''):
        assert self.state == 'body'
        self.message.creationFinished()
        self.messageReceived(self.message)
        self.reset(remainingData)

    def rawDataReceived(self, data):
        assert self.state in ('body', 'invalid')
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        if self.state == 'invalid':
            return
        if self.length == None:
            self.message.bodyDataReceived(data)
        else:
            dataLen = len(data)
            expectedLen = self.length - self.bodyReceived
            if dataLen > expectedLen:
                self.message.bodyDataReceived(data[:expectedLen])
                self.messageDone(data[expectedLen:])
                return
            else:
                self.bodyReceived += dataLen
                self.message.bodyDataReceived(data)
                if self.bodyReceived == self.length:
                    self.messageDone()