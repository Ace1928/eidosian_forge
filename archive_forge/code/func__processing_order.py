import struct
import dns.exception
import dns.immutable
import dns.name
import dns.rdata
import dns.rdtypes.util
@classmethod
def _processing_order(cls, iterable):
    return dns.rdtypes.util.weighted_processing_order(iterable)