import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class ParseType(enum.IntEnum):
    default = 0
    enum = 1
    flags = 2
    datetime = 3
    text = 4
    bytes = 5
    principal_name = 6
    host_address = 7
    token = 8