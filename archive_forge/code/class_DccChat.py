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
class DccChat(basic.LineReceiver, styles.Ephemeral):
    """
    Direct Client Connection protocol type CHAT.

    DCC CHAT is really just your run o' the mill basic.LineReceiver
    protocol.  This class only varies from that slightly, accepting
    either LF or CR LF for a line delimeter for incoming messages
    while always using CR LF for outgoing.

    The lineReceived method implemented here uses the DCC connection's
    'client' attribute (provided upon construction) to deliver incoming
    lines from the DCC chat via IRCClient's normal privmsg interface.
    That's something of a spoof, which you may well want to override.
    """
    queryData = None
    delimiter = CR.encode('ascii') + NL.encode('ascii')
    client = None
    remoteParty = None
    buffer = b''

    def __init__(self, client, queryData=None):
        """
        Initialize a new DCC CHAT session.

        queryData is a 3-tuple of
        (fromUser, targetUserOrChannel, data)
        as received by the CTCP query.

        (To be honest, fromUser is the only thing that's currently
        used here. targetUserOrChannel is potentially useful, while
        the 'data' argument is solely for informational purposes.)
        """
        self.client = client
        if queryData:
            self.queryData = queryData
            self.remoteParty = self.queryData[0]

    def dataReceived(self, data):
        self.buffer = self.buffer + data
        lines = self.buffer.split(LF)
        self.buffer = lines.pop()
        for line in lines:
            if line[-1] == CR:
                line = line[:-1]
            self.lineReceived(line)

    def lineReceived(self, line):
        log.msg(f'DCC CHAT<{self.remoteParty}> {line}')
        self.client.privmsg(self.remoteParty, self.client.nickname, line)