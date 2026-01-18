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
class Record_SRV(tputil.FancyEqMixin, tputil.FancyStrMixin):
    """
    The location of the server(s) for a specific protocol and domain.

    This is an experimental record type.

    @type priority: L{int}
    @ivar priority: The priority of this target host.  A client MUST attempt to
        contact the target host with the lowest-numbered priority it can reach;
        target hosts with the same priority SHOULD be tried in an order defined
        by the weight field.

    @type weight: L{int}
    @ivar weight: Specifies a relative weight for entries with the same
        priority. Larger weights SHOULD be given a proportionately higher
        probability of being selected.

    @type port: L{int}
    @ivar port: The port on this target host of this service.

    @type target: L{Name}
    @ivar target: The domain name of the target host.  There MUST be one or
        more address records for this name, the name MUST NOT be an alias (in
        the sense of RFC 1034 or RFC 2181).  Implementors are urged, but not
        required, to return the address record(s) in the Additional Data
        section.  Unless and until permitted by future standards action, name
        compression is not to be used for this field.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be
        cached.

    @see: U{http://www.faqs.org/rfcs/rfc2782.html}
    """
    TYPE = SRV
    fancybasename = 'SRV'
    compareAttributes = ('priority', 'weight', 'target', 'port', 'ttl')
    showAttributes = ('priority', 'weight', ('target', 'target', '%s'), 'port', 'ttl')

    def __init__(self, priority=0, weight=0, port=0, target=b'', ttl=None):
        """
        @param target: See L{Record_SRV.target}
        @type target: L{bytes} or L{str}
        """
        self.priority = int(priority)
        self.weight = int(weight)
        self.port = int(port)
        self.target = Name(target)
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        strio.write(struct.pack('!HHH', self.priority, self.weight, self.port))
        self.target.encode(strio, None)

    def decode(self, strio, length=None):
        r = struct.unpack('!HHH', readPrecisely(strio, struct.calcsize('!HHH')))
        self.priority, self.weight, self.port = r
        self.target = Name()
        self.target.decode(strio)

    def __hash__(self):
        return hash((self.priority, self.weight, self.port, self.target))