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
def _sendMessage(self, msgType, user, message, length=None):
    """
        Send a message or notice to a user or channel.

        The message will be split into multiple commands to the server if:
         - The message contains any newline characters
         - Any span between newline characters is longer than the given
           line-length.

        @param msgType: Whether a PRIVMSG or NOTICE should be sent.
        @type msgType: C{str}

        @param user: Username or channel name to which to direct the
            message.
        @type user: C{str}

        @param message: Text to send.
        @type message: C{str}

        @param length: Maximum number of octets to send in a single
            command, including the IRC protocol framing. If L{None} is given
            then L{IRCClient._safeMaximumLineLength} is used to determine a
            value.
        @type length: C{int}
        """
    fmt = f'{msgType} {user} :'
    if length is None:
        length = self._safeMaximumLineLength(fmt)
    minimumLength = len(fmt) + 2
    if length <= minimumLength:
        raise ValueError('Maximum length must exceed %d for message to %s' % (minimumLength, user))
    for line in split(message, length - minimumLength):
        self.sendLine(fmt + line)