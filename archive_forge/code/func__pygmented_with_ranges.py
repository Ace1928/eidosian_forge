import ast
import itertools
import types
from collections import OrderedDict, Counter, defaultdict
from types import FrameType, TracebackType
from typing import (
from asttokens import ASTText
def _pygmented_with_ranges(formatter, code, ranges):
    import pygments
    from pygments.lexers import get_lexer_by_name

    class MyLexer(type(get_lexer_by_name('python3'))):

        def get_tokens(self, text):
            length = 0
            for ttype, value in super().get_tokens(text):
                if any((start <= length < end for start, end in ranges)):
                    ttype = ttype.ExecutingNode
                length += len(value)
                yield (ttype, value)
    lexer = MyLexer(stripnl=False)
    try:
        highlighted = pygments.highlight(code, lexer, formatter)
    except Exception:
        highlighted = code
    return highlighted.splitlines()