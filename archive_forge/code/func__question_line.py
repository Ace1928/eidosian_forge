from __future__ import absolute_import
from io import StringIO
import struct
import time
import dns.edns
import dns.exception
import dns.flags
import dns.name
import dns.opcode
import dns.entropy
import dns.rcode
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rrset
import dns.renderer
import dns.tsig
import dns.wiredata
from ._compat import long, xrange, string_types
def _question_line(self, section):
    """Process one line from the text format question section."""
    token = self.tok.get(want_leading=True)
    if not token.is_whitespace():
        self.last_name = dns.name.from_text(token.value, None)
    name = self.last_name
    token = self.tok.get()
    if not token.is_identifier():
        raise dns.exception.SyntaxError
    try:
        rdclass = dns.rdataclass.from_text(token.value)
        token = self.tok.get()
        if not token.is_identifier():
            raise dns.exception.SyntaxError
    except dns.exception.SyntaxError:
        raise dns.exception.SyntaxError
    except Exception:
        rdclass = dns.rdataclass.IN
    rdtype = dns.rdatatype.from_text(token.value)
    self.message.find_rrset(self.message.question, name, rdclass, rdtype, create=True, force_unique=True)
    if self.updating:
        self.zone_rdclass = rdclass
    self.tok.get_eol()