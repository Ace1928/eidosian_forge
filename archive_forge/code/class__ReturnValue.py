import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
@_register
class _ReturnValue(_BaseHandler):
    """
    Returns with TOS to the caller of the function.
    """
    opname = 'RETURN_VALUE'

    def _handle(self):
        v = self.stack.pop()
        return_token = _Token(self.i_line, None, 'return ', end_of_line=True)
        self.tokens.append(return_token)
        for token in v.tokens:
            token.mark_after(return_token)
        self.tokens.extend(v.tokens)
        self._write_tokens()