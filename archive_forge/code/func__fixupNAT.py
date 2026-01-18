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
def _fixupNAT(self, message, sourcePeer):
    srcHost, srcPort = sourcePeer
    senderVia = parseViaHeader(message.headers['via'][0])
    if senderVia.host != srcHost:
        senderVia.received = srcHost
        if senderVia.port != srcPort:
            senderVia.rport = srcPort
        message.headers['via'][0] = senderVia.toString()
    elif senderVia.rport == True:
        senderVia.received = srcHost
        senderVia.rport = srcPort
        message.headers['via'][0] = senderVia.toString()