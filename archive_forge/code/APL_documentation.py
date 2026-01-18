import binascii
import codecs
import struct
import dns.exception
import dns.inet
import dns.rdata
import dns.tokenizer
from dns._compat import xrange, maybe_chr
APL record.

    @ivar items: a list of APL items
    @type items: list of APL_Item
    @see: RFC 3123