from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
class NeedAbsoluteNameOrOrigin(dns.exception.DNSException):
    """An attempt was made to convert a non-absolute name to
    wire when there was also a non-absolute (or missing) origin."""