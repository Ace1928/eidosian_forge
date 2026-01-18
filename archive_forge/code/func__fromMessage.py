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
@classmethod
def _fromMessage(cls, message):
    """
        Construct and return a new L{_EDNSMessage} whose attributes and records
        are derived from the attributes and records of C{message} (a L{Message}
        instance).

        If present, an C{OPT} record will be extracted from the C{additional}
        section and its attributes and options will be used to set the EDNS
        specific attributes C{extendedRCODE}, C{ednsVersion}, C{dnssecOK},
        C{ednsOptions}.

        The C{extendedRCODE} will be combined with C{message.rCode} and assigned
        to C{self.rCode}.

        @param message: The source L{Message}.
        @type message: L{Message}

        @return: A new L{_EDNSMessage}
        @rtype: L{_EDNSMessage}
        """
    additional = []
    optRecords = []
    for r in message.additional:
        if r.type == OPT:
            optRecords.append(_OPTHeader.fromRRHeader(r))
        else:
            additional.append(r)
    newMessage = cls(id=message.id, answer=message.answer, opCode=message.opCode, auth=message.auth, trunc=message.trunc, recDes=message.recDes, recAv=message.recAv, rCode=message.rCode, authenticData=message.authenticData, checkingDisabled=message.checkingDisabled, ednsVersion=None, dnssecOK=False, queries=message.queries[:], answers=message.answers[:], authority=message.authority[:], additional=additional)
    if len(optRecords) == 1:
        opt = optRecords[0]
        newMessage.ednsVersion = opt.version
        newMessage.dnssecOK = opt.dnssecOK
        newMessage.maxSize = opt.udpPayloadSize
        newMessage.rCode = opt.extendedRCODE << 4 | message.rCode
    return newMessage