import errno
import os
import struct
import warnings
from typing import Dict
from zope.interface import implementer
from twisted.conch.interfaces import ISFTPFile, ISFTPServer
from twisted.conch.ssh.common import NS, getNS
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
def _sendStatus(self, requestId, code, message, lang=b''):
    """
        Helper method to send a FXP_STATUS message.
        """
    data = requestId + struct.pack('!L', code)
    data += NS(message)
    data += NS(lang)
    self.sendPacket(FXP_STATUS, data)