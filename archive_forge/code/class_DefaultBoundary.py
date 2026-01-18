import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class DefaultBoundary(ZeroWidthBase):
    _opcode = OP.DEFAULT_BOUNDARY
    _op_name = 'DEFAULT_BOUNDARY'