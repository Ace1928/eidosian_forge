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
class SIPError(Exception):

    def __init__(self, code, phrase=None):
        if phrase is None:
            phrase = statusCodes[code]
        Exception.__init__(self, 'SIP error (%d): %s' % (code, phrase))
        self.code = code
        self.phrase = phrase