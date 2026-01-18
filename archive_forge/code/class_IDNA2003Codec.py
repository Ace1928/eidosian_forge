from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
class IDNA2003Codec(IDNACodec):
    """IDNA 2003 encoder/decoder."""

    def __init__(self, strict_decode=False):
        """Initialize the IDNA 2003 encoder/decoder.

        *strict_decode* is a ``bool``. If `True`, then IDNA2003 checking
        is done when decoding.  This can cause failures if the name
        was encoded with IDNA2008.  The default is `False`.
        """
        super(IDNA2003Codec, self).__init__()
        self.strict_decode = strict_decode

    def encode(self, label):
        """Encode *label*."""
        if label == '':
            return b''
        try:
            return encodings.idna.ToASCII(label)
        except UnicodeError:
            raise LabelTooLong

    def decode(self, label):
        """Decode *label*."""
        if not self.strict_decode:
            return super(IDNA2003Codec, self).decode(label)
        if label == b'':
            return u''
        try:
            return _escapify(encodings.idna.ToUnicode(label), True)
        except Exception as e:
            raise IDNAException(idna_exception=e)