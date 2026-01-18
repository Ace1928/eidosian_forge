import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class FilterBlock(Stmt):
    """Node for filter sections."""
    fields = ('body', 'filter')
    body: t.List[Node]
    filter: 'Filter'