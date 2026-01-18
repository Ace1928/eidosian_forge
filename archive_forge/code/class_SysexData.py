import re
from .checks import check_data, check_msgdict, check_value
from .decode import decode_message
from .encode import encode_message
from .specs import REALTIME_TYPES, SPEC_BY_TYPE, make_msgdict
from .strings import msg2str, str2msg
class SysexData(tuple):
    """Special kind of tuple accepts and converts any sequence in +=."""

    def __iadd__(self, other):
        check_data(other)
        return self + SysexData(other)