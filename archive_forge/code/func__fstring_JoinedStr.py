import ast
import io
import sys
import tokenize
def _fstring_JoinedStr(self, t, write):
    for value in t.values:
        meth = getattr(self, '_fstring_' + type(value).__name__)
        meth(value, write)