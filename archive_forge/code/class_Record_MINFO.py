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
class Record_MINFO(tputil.FancyEqMixin, tputil.FancyStrMixin):
    """
    Mailbox or mail list information.

    This is an experimental record type.

    @type rmailbx: L{Name}
    @ivar rmailbx: A domain-name which specifies a mailbox which is responsible
        for the mailing list or mailbox.  If this domain name names the root,
        the owner of the MINFO RR is responsible for itself.

    @type emailbx: L{Name}
    @ivar emailbx: A domain-name which specifies a mailbox which is to receive
        error messages related to the mailing list or mailbox specified by the
        owner of the MINFO record.  If this domain name names the root, errors
        should be returned to the sender of the message.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be
        cached.
    """
    TYPE = MINFO
    rmailbx = None
    emailbx = None
    fancybasename = 'MINFO'
    compareAttributes = ('rmailbx', 'emailbx', 'ttl')
    showAttributes = (('rmailbx', 'responsibility', '%s'), ('emailbx', 'errors', '%s'), 'ttl')

    def __init__(self, rmailbx=b'', emailbx=b'', ttl=None):
        """
        @param rmailbx: See L{Record_MINFO.rmailbx}.
        @type rmailbx: L{bytes} or L{str}

        @param emailbx: See L{Record_MINFO.rmailbx}.
        @type emailbx: L{bytes} or L{str}
        """
        self.rmailbx, self.emailbx = (Name(rmailbx), Name(emailbx))
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        self.rmailbx.encode(strio, compDict)
        self.emailbx.encode(strio, compDict)

    def decode(self, strio, length=None):
        self.rmailbx, self.emailbx = (Name(), Name())
        self.rmailbx.decode(strio)
        self.emailbx.decode(strio)

    def __hash__(self):
        return hash((self.rmailbx, self.emailbx))