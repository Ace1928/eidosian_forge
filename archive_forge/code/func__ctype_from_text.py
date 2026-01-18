import struct
import base64
import dns.exception
import dns.dnssec
import dns.rdata
import dns.tokenizer
def _ctype_from_text(what):
    v = _ctype_by_name.get(what)
    if v is not None:
        return v
    return int(what)