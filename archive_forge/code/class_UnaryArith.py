import py
import sys, inspect
from compiler import parse, ast, pycodegen
from py._code.assertion import BuiltinAssertionError, _format_explanation
import types
class UnaryArith(Interpretable):
    __view__ = astclass

    def eval(self, frame, astpattern=astpattern):
        expr = Interpretable(self.expr)
        expr.eval(frame)
        self.explanation = astpattern.replace('__exprinfo_expr', expr.explanation)
        try:
            self.result = frame.eval(astpattern, __exprinfo_expr=expr.result)
        except passthroughex:
            raise
        except:
            raise Failure(self)