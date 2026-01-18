import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class EndOfLine(ZeroWidthBase):
    _opcode = OP.END_OF_LINE
    _op_name = 'END_OF_LINE'