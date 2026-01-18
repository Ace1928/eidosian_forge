import py
import sys, inspect
from compiler import parse, ast, pycodegen
from py._code.assertion import BuiltinAssertionError, _format_explanation
import types
class Interpretable(View):
    """A parse tree node with a few extra methods."""
    explanation = None

    def is_builtin(self, frame):
        return False

    def eval(self, frame):
        try:
            expr = ast.Expression(self.__obj__)
            expr.filename = '<eval>'
            self.__obj__.filename = '<eval>'
            co = pycodegen.ExpressionCodeGenerator(expr).getCode()
            result = frame.eval(co)
        except passthroughex:
            raise
        except:
            raise Failure(self)
        self.result = result
        self.explanation = self.explanation or frame.repr(self.result)

    def run(self, frame):
        try:
            expr = ast.Module(None, ast.Stmt([self.__obj__]))
            expr.filename = '<run>'
            co = pycodegen.ModuleCodeGenerator(expr).getCode()
            frame.exec_(co)
        except passthroughex:
            raise
        except:
            raise Failure(self)

    def nice_explanation(self):
        return _format_explanation(self.explanation)