import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
@_register
class _CallFunction(_BaseHandler):
    """

    CALL_FUNCTION(argc)

        Calls a callable object with positional arguments. argc indicates the number of positional
        arguments. The top of the stack contains positional arguments, with the right-most argument
        on top. Below the arguments is a callable object to call. CALL_FUNCTION pops all arguments
        and the callable object off the stack, calls the callable object with those arguments, and
        pushes the return value returned by the callable object.

        Changed in version 3.6: This opcode is used only for calls with positional arguments.

    """
    opname = 'CALL_FUNCTION'

    def _handle(self):
        args = []
        for _i in range(self.instruction.argval + 1):
            arg = self.stack.pop()
            args.append(arg)
        it = reversed(args)
        name = next(it)
        max_line = name.i_line
        for t in name.tokens:
            self.tokens.append(t)
        tok_open_parens = _Token(name.i_line, None, '(', after=name)
        self.tokens.append(tok_open_parens)
        prev = tok_open_parens
        for i, arg in enumerate(it):
            for t in arg.tokens:
                t.mark_after(name)
                t.mark_after(prev)
                max_line = max(max_line, t.i_line)
                self.tokens.append(t)
            prev = arg
            if i > 0:
                comma_token = _Token(prev.i_line, None, ',', after=prev)
                self.tokens.append(comma_token)
                prev = comma_token
        tok_close_parens = _Token(max_line, None, ')', after=prev)
        self.tokens.append(tok_close_parens)
        self._write_tokens()
        self.stack.push(self)