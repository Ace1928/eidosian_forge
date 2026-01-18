import calendar
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from struct import pack, unpack_from
from .exceptions import FrameSyntaxError
from .spec import Basic
from .utils import bytes_to_str as pstr_t
from .utils import str_to_bytes
def _write_array(list_, write, bits):
    out = BytesIO()
    awrite = out.write
    for v in list_:
        try:
            _write_item(v, awrite, bits)
        except ValueError:
            raise FrameSyntaxError(ILLEGAL_TABLE_TYPE_WITH_VALUE.format(type(v), v))
    array_data = out.getvalue()
    write(pack('>I', len(array_data)))
    write(array_data)