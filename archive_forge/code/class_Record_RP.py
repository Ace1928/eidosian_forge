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
@implementer(IEncodableRecord)
class Record_RP(tputil.FancyEqMixin, tputil.FancyStrMixin):
    """
    The responsible person for a domain.

    @type mbox: L{Name}
    @ivar mbox: A domain name that specifies the mailbox for the responsible
        person.

    @type txt: L{Name}
    @ivar txt: A domain name for which TXT RR's exist (indirection through
        which allows information sharing about the contents of this RP record).

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be
        cached.

    @see: U{http://www.faqs.org/rfcs/rfc1183.html}
    """
    TYPE = RP
    fancybasename = 'RP'
    compareAttributes = ('mbox', 'txt', 'ttl')
    showAttributes = (('mbox', 'mbox', '%s'), ('txt', 'txt', '%s'), 'ttl')

    def __init__(self, mbox=b'', txt=b'', ttl=None):
        """
        @param mbox: See L{Record_RP.mbox}.
        @type mbox: L{bytes} or L{str}

        @param txt: See L{Record_RP.txt}
        @type txt: L{bytes} or L{str}
        """
        self.mbox = Name(mbox)
        self.txt = Name(txt)
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        self.mbox.encode(strio, compDict)
        self.txt.encode(strio, compDict)

    def decode(self, strio, length=None):
        self.mbox = Name()
        self.txt = Name()
        self.mbox.decode(strio)
        self.txt.decode(strio)

    def __hash__(self):
        return hash((self.mbox, self.txt))