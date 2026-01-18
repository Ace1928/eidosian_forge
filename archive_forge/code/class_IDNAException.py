from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
class IDNAException(dns.exception.DNSException):
    """IDNA processing raised an exception."""
    supp_kwargs = {'idna_exception'}
    fmt = 'IDNA processing exception: {idna_exception}'