import typing as t
from contextlib import contextmanager
from functools import update_wrapper
from io import StringIO
from itertools import chain
from keyword import iskeyword as is_python_keyword
from markupsafe import escape
from markupsafe import Markup
from . import nodes
from .exceptions import TemplateAssertionError
from .idtracking import Symbols
from .idtracking import VAR_LOAD_ALIAS
from .idtracking import VAR_LOAD_PARAMETER
from .idtracking import VAR_LOAD_RESOLVE
from .idtracking import VAR_LOAD_UNDEFINED
from .nodes import EvalContext
from .optimizer import Optimizer
from .utils import _PassArg
from .utils import concat
from .visitor import NodeVisitor
def _make_binop(op: str) -> t.Callable[['CodeGenerator', nodes.BinExpr, 'Frame'], None]:

    @optimizeconst
    def visitor(self: 'CodeGenerator', node: nodes.BinExpr, frame: Frame) -> None:
        if self.environment.sandboxed and op in self.environment.intercepted_binops:
            self.write(f'environment.call_binop(context, {op!r}, ')
            self.visit(node.left, frame)
            self.write(', ')
            self.visit(node.right, frame)
        else:
            self.write('(')
            self.visit(node.left, frame)
            self.write(f' {op} ')
            self.visit(node.right, frame)
        self.write(')')
    return visitor