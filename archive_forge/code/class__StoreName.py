import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
@_register
class _StoreName(_BaseHandler):
    """
    Implements name = TOS. namei is the index of name in the attribute co_names of the code object.
    The compiler tries to use STORE_FAST or STORE_GLOBAL if possible.
    """
    opname = 'STORE_NAME'

    def _handle(self):
        v = self.stack.pop()
        if isinstance(v, _ForIter):
            v.store_in_name(self)
        elif not isinstance(v, _MakeFunction) or v.is_lambda:
            line = self.i_line
            for t in v.tokens:
                line = min(line, t.i_line)
            t_name = _Token(line, self.instruction)
            t_equal = _Token(line, None, '=', after=t_name)
            self.tokens.append(t_name)
            self.tokens.append(t_equal)
            for t in v.tokens:
                t.mark_after(t_equal)
            self.tokens.extend(v.tokens)
            self._write_tokens()