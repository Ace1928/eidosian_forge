import calendar
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from struct import pack, unpack_from
from .exceptions import FrameSyntaxError
from .spec import Basic
from .utils import bytes_to_str as pstr_t
from .utils import str_to_bytes
def _flushbits(bits, write):
    if bits:
        write(pack('B' * len(bits), *bits))
        bits[:] = []
    return 0