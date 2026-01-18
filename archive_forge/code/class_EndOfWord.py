import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class EndOfWord(ZeroWidthBase):
    _opcode = OP.END_OF_WORD
    _op_name = 'END_OF_WORD'