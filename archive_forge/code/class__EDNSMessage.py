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
class _EDNSMessage(tputil.FancyEqMixin):
    """
    An I{EDNS} message.

    Designed for compatibility with L{Message} but with a narrower public
    interface.

    Most importantly, L{_EDNSMessage.fromStr} will interpret and remove I{OPT}
    records that are present in the additional records section.

    The I{OPT} records are used to populate certain I{EDNS} specific attributes.

    L{_EDNSMessage.toStr} will add suitable I{OPT} records to the additional
    section to represent the extended EDNS information.

    @see: U{https://tools.ietf.org/html/rfc6891}

    @ivar id: See L{__init__}
    @ivar answer: See L{__init__}
    @ivar opCode: See L{__init__}
    @ivar auth: See L{__init__}
    @ivar trunc: See L{__init__}
    @ivar recDes: See L{__init__}
    @ivar recAv: See L{__init__}
    @ivar rCode: See L{__init__}
    @ivar ednsVersion: See L{__init__}
    @ivar dnssecOK: See L{__init__}
    @ivar authenticData: See L{__init__}
    @ivar checkingDisabled: See L{__init__}
    @ivar maxSize: See L{__init__}

    @ivar queries: See L{__init__}
    @ivar answers: See L{__init__}
    @ivar authority: See L{__init__}
    @ivar additional: See L{__init__}

    @ivar _messageFactory: A constructor of L{Message} instances. Called by
        C{_toMessage} and C{_fromMessage}.
    """
    compareAttributes = ('id', 'answer', 'opCode', 'auth', 'trunc', 'recDes', 'recAv', 'rCode', 'ednsVersion', 'dnssecOK', 'authenticData', 'checkingDisabled', 'maxSize', 'queries', 'answers', 'authority', 'additional')
    _messageFactory = Message

    def __init__(self, id=0, answer=False, opCode=OP_QUERY, auth=False, trunc=False, recDes=False, recAv=False, rCode=0, ednsVersion=0, dnssecOK=False, authenticData=False, checkingDisabled=False, maxSize=512, queries=None, answers=None, authority=None, additional=None):
        """
        Construct a new L{_EDNSMessage}

        @see: U{RFC1035 section-4.1.1<https://tools.ietf.org/html/rfc1035#section-4.1.1>}
        @see: U{RFC2535 section-6.1<https://tools.ietf.org/html/rfc2535#section-6.1>}
        @see: U{RFC3225 section-3<https://tools.ietf.org/html/rfc3225#section-3>}
        @see: U{RFC6891 section-6.1.3<https://tools.ietf.org/html/rfc6891#section-6.1.3>}

        @param id: A 16 bit identifier assigned by the program that generates
            any kind of query.  This identifier is copied the corresponding
            reply and can be used by the requester to match up replies to
            outstanding queries.
        @type id: L{int}

        @param answer: A one bit field that specifies whether this message is a
            query (0), or a response (1).
        @type answer: L{bool}

        @param opCode: A four bit field that specifies kind of query in this
            message.  This value is set by the originator of a query and copied
            into the response.
        @type opCode: L{int}

        @param auth: Authoritative Answer - this bit is valid in responses, and
            specifies that the responding name server is an authority for the
            domain name in question section.
        @type auth: L{bool}

        @param trunc: Truncation - specifies that this message was truncated due
            to length greater than that permitted on the transmission channel.
        @type trunc: L{bool}

        @param recDes: Recursion Desired - this bit may be set in a query and is
            copied into the response.  If set, it directs the name server to
            pursue the query recursively. Recursive query support is optional.
        @type recDes: L{bool}

        @param recAv: Recursion Available - this bit is set or cleared in a
            response, and denotes whether recursive query support is available
            in the name server.
        @type recAv: L{bool}

        @param rCode: Extended 12-bit RCODE. Derived from the 4 bits defined in
            U{RFC1035 4.1.1<https://tools.ietf.org/html/rfc1035#section-4.1.1>}
            and the upper 8bits defined in U{RFC6891
            6.1.3<https://tools.ietf.org/html/rfc6891#section-6.1.3>}.
        @type rCode: L{int}

        @param ednsVersion: Indicates the EDNS implementation level. Set to
            L{None} to prevent any EDNS attributes and options being added to
            the encoded byte string.
        @type ednsVersion: L{int} or L{None}

        @param dnssecOK: DNSSEC OK bit as defined by
            U{RFC3225 3<https://tools.ietf.org/html/rfc3225#section-3>}.
        @type dnssecOK: L{bool}

        @param authenticData: A flag indicating in a response that all the data
            included in the answer and authority portion of the response has
            been authenticated by the server according to the policies of that
            server.
            See U{RFC2535 section-6.1<https://tools.ietf.org/html/rfc2535#section-6.1>}.
        @type authenticData: L{bool}

        @param checkingDisabled: A flag indicating in a query that pending
            (non-authenticated) data is acceptable to the resolver sending the
            query.
            See U{RFC2535 section-6.1<https://tools.ietf.org/html/rfc2535#section-6.1>}.
        @type authenticData: L{bool}

        @param maxSize: The requestor's UDP payload size is the number of octets
            of the largest UDP payload that can be reassembled and delivered in
            the requestor's network stack.
        @type maxSize: L{int}

        @param queries: The L{list} of L{Query} associated with this message.
        @type queries: L{list} of L{Query}

        @param answers: The L{list} of answers associated with this message.
        @type answers: L{list} of L{RRHeader}

        @param authority: The L{list} of authority records associated with this
            message.
        @type authority: L{list} of L{RRHeader}

        @param additional: The L{list} of additional records associated with
            this message.
        @type additional: L{list} of L{RRHeader}
        """
        self.id = id
        self.answer = answer
        self.opCode = opCode
        self.auth = auth
        self.trunc = trunc
        self.recDes = recDes
        self.recAv = recAv
        self.rCode = rCode
        self.ednsVersion = ednsVersion
        self.dnssecOK = dnssecOK
        self.authenticData = authenticData
        self.checkingDisabled = checkingDisabled
        self.maxSize = maxSize
        if queries is None:
            queries = []
        self.queries = queries
        if answers is None:
            answers = []
        self.answers = answers
        if authority is None:
            authority = []
        self.authority = authority
        if additional is None:
            additional = []
        self.additional = additional

    def __repr__(self) -> str:
        return _compactRepr(self, flagNames=('answer', 'auth', 'trunc', 'recDes', 'recAv', 'authenticData', 'checkingDisabled', 'dnssecOK'), fieldNames=('id', 'opCode', 'rCode', 'maxSize', 'ednsVersion'), sectionNames=('queries', 'answers', 'authority', 'additional'), alwaysShow=('id',))

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

    def toStr(self):
        """
        Encode to wire format by first converting to a standard L{dns.Message}.

        @return: A L{bytes} string.
        """
        return self._toMessage().toStr()

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

    def fromStr(self, bytes):
        """
        Decode from wire format, saving flags, values and records to this
        L{_EDNSMessage} instance in place.

        @param bytes: The full byte string to be decoded.
        @type bytes: L{bytes}
        """
        m = self._messageFactory()
        m.fromStr(bytes)
        ednsMessage = self._fromMessage(m)
        for attrName in self.compareAttributes:
            setattr(self, attrName, getattr(ednsMessage, attrName))