import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
@_register
class _MakeFunctionPy3(_BaseHandler):
    """
    Pushes a new function object on the stack. From bottom to top, the consumed stack must consist
    of values if the argument carries a specified flag value

        0x01 a tuple of default values for positional-only and positional-or-keyword parameters in positional order

        0x02 a dictionary of keyword-only parameters' default values

        0x04 an annotation dictionary

        0x08 a tuple containing cells for free variables, making a closure

        the code associated with the function (at TOS1)

        the qualified name of the function (at TOS)
    """
    opname = 'MAKE_FUNCTION'
    is_lambda = False

    def _handle(self):
        stack = self.stack
        self.qualified_name = stack.pop()
        self.code = stack.pop()
        default_node = None
        if self.instruction.argval & 1:
            default_node = stack.pop()
        is_lambda = self.is_lambda = '<lambda>' in [x.tok for x in self.qualified_name.tokens]
        if not is_lambda:
            def_token = _Token(self.i_line, None, 'def ')
            self.tokens.append(def_token)
        for token in self.qualified_name.tokens:
            self.tokens.append(token)
            if not is_lambda:
                token.mark_after(def_token)
        prev = token
        open_parens_token = _Token(self.i_line, None, '(', after=prev)
        self.tokens.append(open_parens_token)
        prev = open_parens_token
        code = self.code.instruction.argval
        if default_node:
            defaults = [_SENTINEL] * (len(code.co_varnames) - len(default_node.instruction.argval)) + list(default_node.instruction.argval)
        else:
            defaults = [_SENTINEL] * len(code.co_varnames)
        for i, arg in enumerate(code.co_varnames):
            if i > 0:
                comma_token = _Token(prev.i_line, None, ', ', after=prev)
                self.tokens.append(comma_token)
                prev = comma_token
            arg_token = _Token(self.i_line, None, arg, after=prev)
            self.tokens.append(arg_token)
            default = defaults[i]
            if default is not _SENTINEL:
                eq_token = _Token(default_node.i_line, None, '=', after=prev)
                self.tokens.append(eq_token)
                prev = eq_token
                default_token = _Token(default_node.i_line, None, str(default), after=prev)
                self.tokens.append(default_token)
                prev = default_token
        tok_close_parens = _Token(prev.i_line, None, '):', after=prev)
        self.tokens.append(tok_close_parens)
        self._write_tokens()
        stack.push(self)
        self.writer.indent(prev.i_line + 1)
        self.writer.dedent(max(self.disassembler.merge_code(code)))