import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class FromImport(Stmt):
    """A node that represents the from import tag.  It's important to not
    pass unsafe names to the name attribute.  The compiler translates the
    attribute lookups directly into getattr calls and does *not* use the
    subscript callback of the interface.  As exported variables may not
    start with double underscores (which the parser asserts) this is not a
    problem for regular Jinja code, but if this node is used in an extension
    extra care must be taken.

    The list of names may contain tuples if aliases are wanted.
    """
    fields = ('template', 'names', 'with_context')
    template: 'Expr'
    names: t.List[t.Union[str, t.Tuple[str, str]]]
    with_context: bool