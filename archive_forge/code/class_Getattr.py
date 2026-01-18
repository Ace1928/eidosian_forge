import py
import sys, inspect
from compiler import parse, ast, pycodegen
from py._code.assertion import BuiltinAssertionError, _format_explanation
import types
class Getattr(Interpretable):
    __view__ = ast.Getattr

    def eval(self, frame):
        expr = Interpretable(self.expr)
        expr.eval(frame)
        source = '__exprinfo_expr.%s' % self.attrname
        try:
            self.result = frame.eval(source, __exprinfo_expr=expr.result)
        except passthroughex:
            raise
        except:
            raise Failure(self)
        self.explanation = '%s.%s' % (expr.explanation, self.attrname)
        source = 'hasattr(__exprinfo_expr, "__dict__") and %r in __exprinfo_expr.__dict__' % self.attrname
        try:
            from_instance = frame.is_true(frame.eval(source, __exprinfo_expr=expr.result))
        except passthroughex:
            raise
        except:
            from_instance = True
        if from_instance:
            r = frame.repr(self.result)
            self.explanation = '%s\n{%s = %s\n}' % (r, r, self.explanation)