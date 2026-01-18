import struct
import dns.exception
import dns.rdata
import dns.tokenizer
from dns._compat import long, text_type
def _validate_float_string(what):
    if what[0] == b'-'[0] or what[0] == b'+'[0]:
        what = what[1:]
    if what.isdigit():
        return
    left, right = what.split(b'.')
    if left == b'' and right == b'':
        raise dns.exception.FormError
    if not left == b'' and (not left.decode().isdigit()):
        raise dns.exception.FormError
    if not right == b'' and (not right.decode().isdigit()):
        raise dns.exception.FormError