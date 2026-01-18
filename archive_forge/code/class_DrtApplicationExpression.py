import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class DrtApplicationExpression(DrtExpression, ApplicationExpression):

    def fol(self):
        return ApplicationExpression(self.function.fol(), self.argument.fol())

    def get_refs(self, recursive=False):
        """:see: AbstractExpression.get_refs()"""
        return self.function.get_refs(True) + self.argument.get_refs(True) if recursive else []

    def _pretty(self):
        function, args = self.uncurry()
        function_lines = function._pretty()
        args_lines = [arg._pretty() for arg in args]
        max_lines = max(map(len, [function_lines] + args_lines))
        function_lines = _pad_vertically(function_lines, max_lines)
        args_lines = [_pad_vertically(arg_lines, max_lines) for arg_lines in args_lines]
        func_args_lines = list(zip(function_lines, list(zip(*args_lines))))
        return [func_line + ' ' + ' '.join(args_line) + ' ' for func_line, args_line in func_args_lines[:2]] + [func_line + '(' + ','.join(args_line) + ')' for func_line, args_line in func_args_lines[2:3]] + [func_line + ' ' + ' '.join(args_line) + ' ' for func_line, args_line in func_args_lines[3:]]