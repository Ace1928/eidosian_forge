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
def _cbUnregister(self, registration, message):
    msg = self.responseFromRequest(200, message)
    msg.headers.setdefault('contact', []).append(registration.contactURL.toString())
    msg.addHeader('expires', '0')
    self.deliverResponse(msg)