import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class AnyAll(Any):
    _opcode = {False: OP.ANY_ALL, True: OP.ANY_ALL_REV}
    _op_name = 'ANY_ALL'