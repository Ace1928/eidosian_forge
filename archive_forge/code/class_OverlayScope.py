import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class OverlayScope(Stmt):
    """An overlay scope for extensions.  This is a largely unoptimized scope
    that however can be used to introduce completely arbitrary variables into
    a sub scope from a dictionary or dictionary like object.  The `context`
    field has to evaluate to a dictionary object.

    Example usage::

        OverlayScope(context=self.call_method('get_context'),
                     body=[...])

    .. versionadded:: 2.10
    """
    fields = ('context', 'body')
    context: Expr
    body: t.List[Node]