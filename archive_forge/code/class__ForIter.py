import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
@_register
class _ForIter(_BaseHandler):
    """
    TOS is an iterator. Call its __next__() method. If this yields a new value, push it on the stack
    (leaving the iterator below it). If the iterator indicates it is exhausted TOS is popped, and
    the byte code counter is incremented by delta.
    """
    opname = 'FOR_ITER'
    iter_in = None

    def _handle(self):
        self.iter_in = self.stack.pop()
        self.stack.push(self)

    def store_in_name(self, store_name):
        for_token = _Token(self.i_line, None, 'for ')
        self.tokens.append(for_token)
        prev = for_token
        t_name = _Token(store_name.i_line, store_name.instruction, after=prev)
        self.tokens.append(t_name)
        prev = t_name
        in_token = _Token(store_name.i_line, None, ' in ', after=prev)
        self.tokens.append(in_token)
        prev = in_token
        max_line = store_name.i_line
        if self.iter_in:
            for t in self.iter_in.tokens:
                t.mark_after(prev)
                max_line = max(max_line, t.i_line)
                prev = t
            self.tokens.extend(self.iter_in.tokens)
        colon_token = _Token(self.i_line, None, ':', after=prev)
        self.tokens.append(colon_token)
        prev = for_token
        self._write_tokens()