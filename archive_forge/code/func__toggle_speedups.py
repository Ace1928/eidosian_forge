from __future__ import absolute_import
from decimal import Decimal
from .errors import JSONDecodeError
from .raw_json import RawJSON
from .decoder import JSONDecoder
from .encoder import JSONEncoder, JSONEncoderForHTML
def _toggle_speedups(enabled):
    from . import decoder as dec
    from . import encoder as enc
    from . import scanner as scan
    c_make_encoder = _import_c_make_encoder()
    if enabled:
        dec.scanstring = dec.c_scanstring or dec.py_scanstring
        enc.c_make_encoder = c_make_encoder
        enc.encode_basestring_ascii = enc.c_encode_basestring_ascii or enc.py_encode_basestring_ascii
        scan.make_scanner = scan.c_make_scanner or scan.py_make_scanner
    else:
        dec.scanstring = dec.py_scanstring
        enc.c_make_encoder = None
        enc.encode_basestring_ascii = enc.py_encode_basestring_ascii
        scan.make_scanner = scan.py_make_scanner
    dec.make_scanner = scan.make_scanner
    global _default_decoder
    _default_decoder = JSONDecoder()
    global _default_encoder
    _default_encoder = JSONEncoder()