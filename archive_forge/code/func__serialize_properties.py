import calendar
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from struct import pack, unpack_from
from .exceptions import FrameSyntaxError
from .spec import Basic
from .utils import bytes_to_str as pstr_t
from .utils import str_to_bytes
def _serialize_properties(self):
    """Serialize AMQP properties.

        Serialize the 'properties' attribute (a dictionary) into
        the raw bytes making up a set of property flags and a
        property list, suitable for putting into a content frame header.
        """
    shift = 15
    flag_bits = 0
    flags = []
    sformat, svalues = ([], [])
    props = self.properties
    for key, proptype in self.PROPERTIES:
        val = props.get(key, None)
        if val is not None:
            if shift == 0:
                flags.append(flag_bits)
                flag_bits = 0
                shift = 15
            flag_bits |= 1 << shift
            if proptype != 'bit':
                sformat.append(str_to_bytes(proptype))
                svalues.append(val)
        shift -= 1
    flags.append(flag_bits)
    result = BytesIO()
    write = result.write
    for flag_bits in flags:
        write(pack('>H', flag_bits))
    write(dumps(b''.join(sformat), svalues))
    return result.getvalue()