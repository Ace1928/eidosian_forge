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
def _toMessage(self):
    """
        Convert to a standard L{dns.Message}.

        If C{ednsVersion} is not None, an L{_OPTHeader} instance containing all
        the I{EDNS} specific attributes and options will be appended to the list
        of C{additional} records.

        @return: A L{dns.Message}
        @rtype: L{dns.Message}
        """
    m = self._messageFactory(id=self.id, answer=self.answer, opCode=self.opCode, auth=self.auth, trunc=self.trunc, recDes=self.recDes, recAv=self.recAv, rCode=self.rCode & 15, authenticData=self.authenticData, checkingDisabled=self.checkingDisabled)
    m.queries = self.queries[:]
    m.answers = self.answers[:]
    m.authority = self.authority[:]
    m.additional = self.additional[:]
    if self.ednsVersion is not None:
        o = _OPTHeader(version=self.ednsVersion, dnssecOK=self.dnssecOK, udpPayloadSize=self.maxSize, extendedRCODE=self.rCode >> 4)
        m.additional.append(o)
    return m