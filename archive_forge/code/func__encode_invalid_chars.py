from __future__ import absolute_import
import re
from collections import namedtuple
from ..exceptions import LocationParseError
from ..packages import six
def _encode_invalid_chars(component, allowed_chars, encoding='utf-8'):
    """Percent-encodes a URI component without reapplying
    onto an already percent-encoded component.
    """
    if component is None:
        return component
    component = six.ensure_text(component)
    component, percent_encodings = PERCENT_RE.subn(lambda match: match.group(0).upper(), component)
    uri_bytes = component.encode('utf-8', 'surrogatepass')
    is_percent_encoded = percent_encodings == uri_bytes.count(b'%')
    encoded_component = bytearray()
    for i in range(0, len(uri_bytes)):
        byte = uri_bytes[i:i + 1]
        byte_ord = ord(byte)
        if is_percent_encoded and byte == b'%' or (byte_ord < 128 and byte.decode() in allowed_chars):
            encoded_component += byte
            continue
        encoded_component.extend(b'%' + hex(byte_ord)[2:].encode().zfill(2).upper())
    return encoded_component.decode(encoding)