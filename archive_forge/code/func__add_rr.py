import dns.message
import dns.name
import dns.opcode
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.tsig
from ._compat import string_types
def _add_rr(self, name, ttl, rd, deleting=None, section=None):
    """Add a single RR to the update section."""
    if section is None:
        section = self.authority
    covers = rd.covers()
    rrset = self.find_rrset(section, name, self.zone_rdclass, rd.rdtype, covers, deleting, True, True)
    rrset.add(rd, ttl)