import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def _decode_uu(encoded):
    """Decode uuencoded data."""
    decoded_lines = []
    encoded_lines_iter = iter(encoded.splitlines())
    for line in encoded_lines_iter:
        if line.startswith(b'begin '):
            mode, _, path = line.removeprefix(b'begin ').partition(b' ')
            try:
                int(mode, base=8)
            except ValueError:
                continue
            else:
                break
    else:
        raise ValueError('`begin` line not found')
    for line in encoded_lines_iter:
        if not line:
            raise ValueError('Truncated input')
        elif line.strip(b' \t\r\n\x0c') == b'end':
            break
        try:
            decoded_line = binascii.a2b_uu(line)
        except binascii.Error:
            nbytes = ((line[0] - 32 & 63) * 4 + 5) // 3
            decoded_line = binascii.a2b_uu(line[:nbytes])
        decoded_lines.append(decoded_line)
    return b''.join(decoded_lines)