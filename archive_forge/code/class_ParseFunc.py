import operator
import _ast
from mako import _ast_util
from mako import compat
from mako import exceptions
from mako import util
class ParseFunc(_ast_util.NodeVisitor):

    def __init__(self, listener, **exception_kwargs):
        self.listener = listener
        self.exception_kwargs = exception_kwargs

    def visit_FunctionDef(self, node):
        self.listener.funcname = node.name
        argnames = [arg_id(arg) for arg in node.args.args]
        if node.args.vararg:
            argnames.append(node.args.vararg.arg)
        kwargnames = [arg_id(arg) for arg in node.args.kwonlyargs]
        if node.args.kwarg:
            kwargnames.append(node.args.kwarg.arg)
        self.listener.argnames = argnames
        self.listener.defaults = node.args.defaults
        self.listener.kwargnames = kwargnames
        self.listener.kwdefaults = node.args.kw_defaults
        self.listener.varargs = node.args.vararg
        self.listener.kwargs = node.args.kwarg