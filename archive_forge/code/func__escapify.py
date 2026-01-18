from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
def _escapify(label, unicode_mode=False):
    """Escape the characters in label which need it.
    @param unicode_mode: escapify only special and whitespace (<= 0x20)
    characters
    @returns: the escaped string
    @rtype: string"""
    if not unicode_mode:
        text = ''
        if isinstance(label, text_type):
            label = label.encode()
        for c in bytearray(label):
            if c in _escaped:
                text += '\\' + chr(c)
            elif c > 32 and c < 127:
                text += chr(c)
            else:
                text += '\\%03d' % c
        return text.encode()
    text = u''
    if isinstance(label, binary_type):
        label = label.decode()
    for c in label:
        if c > u' ' and c < u'\x7f':
            text += c
        elif c >= u'\x7f':
            text += c
        else:
            text += u'\\%03d' % ord(c)
    return text