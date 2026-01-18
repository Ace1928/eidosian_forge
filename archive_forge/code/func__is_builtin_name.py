import sys
import ast
import py
from py._code.assertion import _format_explanation, BuiltinAssertionError
def _is_builtin_name(self, name):
    pattern = '%r not in globals() and %r not in locals()'
    source = pattern % (name.id, name.id)
    co = self._compile(source)
    try:
        return self.frame.eval(co)
    except Exception:
        return False