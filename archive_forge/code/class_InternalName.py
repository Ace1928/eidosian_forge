import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class InternalName(Expr):
    """An internal name in the compiler.  You cannot create these nodes
    yourself but the parser provides a
    :meth:`~jinja2.parser.Parser.free_identifier` method that creates
    a new identifier for you.  This identifier is not available from the
    template and is not treated specially by the compiler.
    """
    fields = ('name',)
    name: str

    def __init__(self) -> None:
        raise TypeError("Can't create internal names.  Use the `free_identifier` method on a parser.")