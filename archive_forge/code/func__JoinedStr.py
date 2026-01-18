import ast
import io
import sys
import tokenize
def _JoinedStr(self, t):
    self.write('f')
    string = io.StringIO()
    self._fstring_JoinedStr(t, string.write)
    self.write(repr(string.getvalue()))