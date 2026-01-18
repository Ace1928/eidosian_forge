from io import BytesIO
import struct
import time
import dns.exception
import dns.name
import dns.node
import dns.rdataset
import dns.rdata
import dns.rdatatype
import dns.rdataclass
from ._compat import string_types
def _make_algorithm_id(algorithm):
    if _is_md5(algorithm):
        oid = [42, 134, 72, 134, 247, 13, 2, 5]
    elif _is_sha1(algorithm):
        oid = [43, 14, 3, 2, 26]
    elif _is_sha256(algorithm):
        oid = [96, 134, 72, 1, 101, 3, 4, 2, 1]
    elif _is_sha512(algorithm):
        oid = [96, 134, 72, 1, 101, 3, 4, 2, 3]
    else:
        raise ValidationFailure('unknown algorithm %u' % algorithm)
    olen = len(oid)
    dlen = _make_hash(algorithm).digest_size
    idbytes = [48] + [8 + olen + dlen] + [48, olen + 4] + [6, olen] + oid + [5, 0] + [4, dlen]
    return struct.pack('!%dB' % len(idbytes), *idbytes)