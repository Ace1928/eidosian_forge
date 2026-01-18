import struct
import dns.exception
import dns.immutable
import dns.name
import dns.rdata
import dns.rdatatype
import dns.rdtypes.util
@dns.immutable.immutable
class Bitmap(dns.rdtypes.util.Bitmap):
    type_name = 'CSYNC'