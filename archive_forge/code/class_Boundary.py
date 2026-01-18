import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class Boundary(ZeroWidthBase):
    _opcode = OP.BOUNDARY
    _op_name = 'BOUNDARY'