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
class DependencyFinderVisitor(NodeVisitor):
    """A visitor that collects filter and test calls."""

    def __init__(self) -> None:
        self.filters: t.Set[str] = set()
        self.tests: t.Set[str] = set()

    def visit_Filter(self, node: nodes.Filter) -> None:
        self.generic_visit(node)
        self.filters.add(node.name)

    def visit_Test(self, node: nodes.Test) -> None:
        self.generic_visit(node)
        self.tests.add(node.name)

    def visit_Block(self, node: nodes.Block) -> None:
        """Stop visiting at blocks."""