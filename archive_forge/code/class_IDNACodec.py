from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
class IDNACodec(object):
    """Abstract base class for IDNA encoder/decoders."""

    def __init__(self):
        pass

    def encode(self, label):
        raise NotImplementedError

    def decode(self, label):
        downcased = label.lower()
        if downcased.startswith(b'xn--'):
            try:
                label = downcased[4:].decode('punycode')
            except Exception as e:
                raise IDNAException(idna_exception=e)
        else:
            label = maybe_decode(label)
        return _escapify(label, True)