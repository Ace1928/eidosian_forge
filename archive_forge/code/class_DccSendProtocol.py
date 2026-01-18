import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
class DccSendProtocol(protocol.Protocol, styles.Ephemeral):
    """
    Protocol for an outgoing Direct Client Connection SEND.

    @ivar blocksize: An integer representing the size of an individual block of
        data.
    @type blocksize: L{int}

    @ivar file: The file to be sent.  This can be either a file object or
        simply the name of the file.
    @type file: L{file} or L{bytes}

    @ivar bytesSent: An integer representing the number of bytes sent.
    @type bytesSent: L{int}

    @ivar completed: An integer representing whether the transfer has been
        completed or not.
    @type completed: L{int}

    @ivar connected: An integer representing whether the connection has been
        established or not.
    @type connected: L{int}
    """
    blocksize = 1024
    file = None
    bytesSent = 0
    completed = 0
    connected = 0

    def __init__(self, file):
        if type(file) is str:
            self.file = open(file)

    def connectionMade(self):
        self.connected = 1
        self.sendBlock()

    def dataReceived(self, data):
        bytesShesGot = struct.unpack('!I', data)
        if bytesShesGot < self.bytesSent:
            return
        elif bytesShesGot > self.bytesSent:
            self.transport.loseConnection()
            return
        self.sendBlock()

    def sendBlock(self):
        block = self.file.read(self.blocksize)
        if block:
            self.transport.write(block)
            self.bytesSent = self.bytesSent + len(block)
        else:
            self.transport.loseConnection()
            self.completed = 1

    def connectionLost(self, reason):
        self.connected = 0
        if hasattr(self.file, 'close'):
            self.file.close()