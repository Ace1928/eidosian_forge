import hashlib
import hmac
import struct
import dns.exception
import dns.rdataclass
import dns.name
from ._compat import long, string_types, text_type
class PeerBadTruncation(PeerError):
    """The peer didn't like amount of truncation in the TSIG we sent"""