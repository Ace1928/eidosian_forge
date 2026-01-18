import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime
def _encode_base64(data, max_line_length):
    encoded_lines = []
    unencoded_bytes_per_line = max_line_length // 4 * 3
    for i in range(0, len(data), unencoded_bytes_per_line):
        thisline = data[i:i + unencoded_bytes_per_line]
        encoded_lines.append(binascii.b2a_base64(thisline).decode('ascii'))
    return ''.join(encoded_lines)