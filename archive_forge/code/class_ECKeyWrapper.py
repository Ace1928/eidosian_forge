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
class ECKeyWrapper(object):

    def __init__(self, key, key_len):
        self.key = key
        self.key_len = key_len

    def verify(self, digest, sig):
        diglong = number.bytes_to_long(digest)
        return self.key.pubkey.verifies(diglong, sig)