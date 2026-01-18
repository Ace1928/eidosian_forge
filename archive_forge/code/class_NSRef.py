import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class NSRef(Expr):
    """Reference to a namespace value assignment"""
    fields = ('name', 'attr')
    name: str
    attr: str

    def can_assign(self) -> bool:
        return True