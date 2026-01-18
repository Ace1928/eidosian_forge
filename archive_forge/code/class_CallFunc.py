import py
import sys, inspect
from compiler import parse, ast, pycodegen
from py._code.assertion import BuiltinAssertionError, _format_explanation
import types
class CallFunc(Interpretable):
    __view__ = ast.CallFunc

    def is_bool(self, frame):
        source = 'isinstance(__exprinfo_value, bool)'
        try:
            return frame.is_true(frame.eval(source, __exprinfo_value=self.result))
        except passthroughex:
            raise
        except:
            return False

    def eval(self, frame):
        node = Interpretable(self.node)
        node.eval(frame)
        explanations = []
        vars = {'__exprinfo_fn': node.result}
        source = '__exprinfo_fn('
        for a in self.args:
            if isinstance(a, ast.Keyword):
                keyword = a.name
                a = a.expr
            else:
                keyword = None
            a = Interpretable(a)
            a.eval(frame)
            argname = '__exprinfo_%d' % len(vars)
            vars[argname] = a.result
            if keyword is None:
                source += argname + ','
                explanations.append(a.explanation)
            else:
                source += '%s=%s,' % (keyword, argname)
                explanations.append('%s=%s' % (keyword, a.explanation))
        if self.star_args:
            star_args = Interpretable(self.star_args)
            star_args.eval(frame)
            argname = '__exprinfo_star'
            vars[argname] = star_args.result
            source += '*' + argname + ','
            explanations.append('*' + star_args.explanation)
        if self.dstar_args:
            dstar_args = Interpretable(self.dstar_args)
            dstar_args.eval(frame)
            argname = '__exprinfo_kwds'
            vars[argname] = dstar_args.result
            source += '**' + argname + ','
            explanations.append('**' + dstar_args.explanation)
        self.explanation = '%s(%s)' % (node.explanation, ', '.join(explanations))
        if source.endswith(','):
            source = source[:-1]
        source += ')'
        try:
            self.result = frame.eval(source, **vars)
        except passthroughex:
            raise
        except:
            raise Failure(self)
        if not node.is_builtin(frame) or not self.is_bool(frame):
            r = frame.repr(self.result)
            self.explanation = '%s\n{%s = %s\n}' % (r, r, self.explanation)