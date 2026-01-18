import math
import re
from typing import (
import unicodedata
from .parser import Parser
def _is_reserved_word(k):
    global _reserved_word_re
    if _reserved_word_re is None:
        _reserved_word_re = re.compile('(' + '|'.join(['break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else', 'enum', 'export', 'extends', 'false', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'new', 'null', 'return', 'super', 'switch', 'this', 'throw', 'true', 'try', 'typeof', 'var', 'void', 'while', 'with']) + ')$')
    return _reserved_word_re.match(k) is not None