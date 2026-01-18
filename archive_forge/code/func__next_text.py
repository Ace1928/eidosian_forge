import base64
import binascii
import struct
import dns.exception
import dns.immutable
import dns.rdata
import dns.rdatatype
import dns.rdtypes.util
def _next_text(self):
    next = base64.b32encode(self.next).translate(b32_normal_to_hex).lower().decode()
    next = next.rstrip('=')
    return next