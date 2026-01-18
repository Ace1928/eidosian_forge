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
class UndeclaredNameVisitor(NodeVisitor):
    """A visitor that checks if a name is accessed without being
    declared.  This is different from the frame visitor as it will
    not stop at closure frames.
    """

    def __init__(self, names: t.Iterable[str]) -> None:
        self.names = set(names)
        self.undeclared: t.Set[str] = set()

    def visit_Name(self, node: nodes.Name) -> None:
        if node.ctx == 'load' and node.name in self.names:
            self.undeclared.add(node.name)
            if self.undeclared == self.names:
                raise VisitorExit()
        else:
            self.names.discard(node.name)

    def visit_Block(self, node: nodes.Block) -> None:
        """Stop visiting a blocks."""