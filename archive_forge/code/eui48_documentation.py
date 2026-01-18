import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (

    :param packed_int: a packed string containing an unsigned integer.
        It is assumed that string is packed in network byte order.

    :return: An unsigned integer equivalent to value of network address
        represented by packed binary string.
    