from __future__ import generators
import sys
import re
import os
from io import BytesIO
import dns.exception
import dns.name
import dns.node
import dns.rdataclass
import dns.rdatatype
import dns.rdata
import dns.rdtypes.ANY.SOA
import dns.rrset
import dns.tokenizer
import dns.ttl
import dns.grange
from ._compat import string_types, text_type, PY3
def _rr_line(self):
    """Process one line from a DNS master file."""
    if self.current_origin is None:
        raise UnknownOrigin
    token = self.tok.get(want_leading=True)
    if not token.is_whitespace():
        self.last_name = dns.name.from_text(token.value, self.current_origin)
    else:
        token = self.tok.get()
        if token.is_eol_or_eof():
            return
        self.tok.unget(token)
    name = self.last_name
    if not name.is_subdomain(self.zone.origin):
        self._eat_line()
        return
    if self.relativize:
        name = name.relativize(self.zone.origin)
    token = self.tok.get()
    if not token.is_identifier():
        raise dns.exception.SyntaxError
    try:
        ttl = dns.ttl.from_text(token.value)
        self.last_ttl = ttl
        self.last_ttl_known = True
        token = self.tok.get()
        if not token.is_identifier():
            raise dns.exception.SyntaxError
    except dns.ttl.BadTTL:
        if not (self.last_ttl_known or self.default_ttl_known):
            raise dns.exception.SyntaxError('Missing default TTL value')
        if self.default_ttl_known:
            ttl = self.default_ttl
        else:
            ttl = self.last_ttl
    try:
        rdclass = dns.rdataclass.from_text(token.value)
        token = self.tok.get()
        if not token.is_identifier():
            raise dns.exception.SyntaxError
    except dns.exception.SyntaxError:
        raise dns.exception.SyntaxError
    except Exception:
        rdclass = self.zone.rdclass
    if rdclass != self.zone.rdclass:
        raise dns.exception.SyntaxError("RR class is not zone's class")
    try:
        rdtype = dns.rdatatype.from_text(token.value)
    except:
        raise dns.exception.SyntaxError("unknown rdatatype '%s'" % token.value)
    n = self.zone.nodes.get(name)
    if n is None:
        n = self.zone.node_factory()
        self.zone.nodes[name] = n
    try:
        rd = dns.rdata.from_text(rdclass, rdtype, self.tok, self.current_origin, False)
    except dns.exception.SyntaxError:
        ty, va = sys.exc_info()[:2]
        raise va
    except:
        ty, va = sys.exc_info()[:2]
        raise dns.exception.SyntaxError('caught exception {}: {}'.format(str(ty), str(va)))
    if not self.default_ttl_known and isinstance(rd, dns.rdtypes.ANY.SOA.SOA):
        self.default_ttl = rd.minimum
        self.default_ttl_known = True
    rd.choose_relativity(self.zone.origin, self.relativize)
    covers = rd.covers()
    rds = n.find_rdataset(rdclass, rdtype, covers, True)
    rds.add(rd, ttl)