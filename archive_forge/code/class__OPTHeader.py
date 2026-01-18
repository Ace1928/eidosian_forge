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
@implementer(IEncodable)
class _OPTHeader(tputil.FancyStrMixin, tputil.FancyEqMixin):
    """
    An OPT record header.

    @ivar name: The DNS name associated with this record. Since this
        is a pseudo record, the name is always an L{Name} instance
        with value b'', which represents the DNS root zone. This
        attribute is a readonly property.

    @ivar type: The DNS record type. This is a fixed value of 41
        C{dns.OPT} for OPT Record. This attribute is a readonly
        property.

    @see: L{_OPTHeader.__init__} for documentation of other public
        instance attributes.

    @see: U{https://tools.ietf.org/html/rfc6891#section-6.1.2}

    @since: 13.2
    """
    showAttributes = (('name', lambda n: nativeString(n.name)), 'type', 'udpPayloadSize', 'extendedRCODE', 'version', 'dnssecOK', 'options')
    compareAttributes = ('name', 'type', 'udpPayloadSize', 'extendedRCODE', 'version', 'dnssecOK', 'options')

    def __init__(self, udpPayloadSize=4096, extendedRCODE=0, version=0, dnssecOK=False, options=None):
        """
        @type udpPayloadSize: L{int}
        @param udpPayloadSize: The number of octets of the largest UDP
            payload that can be reassembled and delivered in the
            requestor's network stack.

        @type extendedRCODE: L{int}
        @param extendedRCODE: Forms the upper 8 bits of extended
            12-bit RCODE (together with the 4 bits defined in
            [RFC1035].  Note that EXTENDED-RCODE value 0 indicates
            that an unextended RCODE is in use (values 0 through 15).

        @type version: L{int}
        @param version: Indicates the implementation level of the
            setter.  Full conformance with this specification is
            indicated by version C{0}.

        @type dnssecOK: L{bool}
        @param dnssecOK: DNSSEC OK bit as defined by [RFC3225].

        @type options: L{list}
        @param options: A L{list} of 0 or more L{_OPTVariableOption}
            instances.
        """
        self.udpPayloadSize = udpPayloadSize
        self.extendedRCODE = extendedRCODE
        self.version = version
        self.dnssecOK = dnssecOK
        if options is None:
            options = []
        self.options = options

    @property
    def name(self):
        """
        A readonly property for accessing the C{name} attribute of
        this record.

        @return: The DNS name associated with this record. Since this
            is a pseudo record, the name is always an L{Name} instance
            with value b'', which represents the DNS root zone.
        """
        return Name(b'')

    @property
    def type(self):
        """
        A readonly property for accessing the C{type} attribute of
        this record.

        @return: The DNS record type. This is a fixed value of 41
            (C{dns.OPT} for OPT Record.
        """
        return OPT

    def encode(self, strio, compDict=None):
        """
        Encode this L{_OPTHeader} instance to bytes.

        @type strio: file
        @param strio: the byte representation of this L{_OPTHeader}
            will be written to this file.

        @type compDict: L{dict} or L{None}
        @param compDict: A dictionary of backreference addresses that
            have already been written to this stream and that may
            be used for DNS name compression.
        """
        b = BytesIO()
        for o in self.options:
            o.encode(b)
        optionBytes = b.getvalue()
        RRHeader(name=self.name.name, type=self.type, cls=self.udpPayloadSize, ttl=self.extendedRCODE << 24 | self.version << 16 | self.dnssecOK << 15, payload=UnknownRecord(optionBytes)).encode(strio, compDict)

    def decode(self, strio, length=None):
        """
        Decode bytes into an L{_OPTHeader} instance.

        @type strio: file
        @param strio: Bytes will be read from this file until the full
            L{_OPTHeader} is decoded.

        @type length: L{int} or L{None}
        @param length: Not used.
        """
        h = RRHeader()
        h.decode(strio, length)
        h.payload = UnknownRecord(readPrecisely(strio, h.rdlength))
        newOptHeader = self.fromRRHeader(h)
        for attrName in self.compareAttributes:
            if attrName not in ('name', 'type'):
                setattr(self, attrName, getattr(newOptHeader, attrName))

    @classmethod
    def fromRRHeader(cls, rrHeader):
        """
        A classmethod for constructing a new L{_OPTHeader} from the
        attributes and payload of an existing L{RRHeader} instance.

        @type rrHeader: L{RRHeader}
        @param rrHeader: An L{RRHeader} instance containing an
            L{UnknownRecord} payload.

        @return: An instance of L{_OPTHeader}.
        @rtype: L{_OPTHeader}
        """
        options = None
        if rrHeader.payload is not None:
            options = []
            optionsBytes = BytesIO(rrHeader.payload.data)
            optionsBytesLength = len(rrHeader.payload.data)
            while optionsBytes.tell() < optionsBytesLength:
                o = _OPTVariableOption()
                o.decode(optionsBytes)
                options.append(o)
        return cls(udpPayloadSize=rrHeader.cls, extendedRCODE=rrHeader.ttl >> 24, version=rrHeader.ttl >> 16 & 255, dnssecOK=(rrHeader.ttl & 65535) >> 15, options=options)