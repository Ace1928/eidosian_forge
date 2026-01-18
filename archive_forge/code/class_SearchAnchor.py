import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class SearchAnchor(ZeroWidthBase):
    _opcode = OP.SEARCH_ANCHOR
    _op_name = 'SEARCH_ANCHOR'