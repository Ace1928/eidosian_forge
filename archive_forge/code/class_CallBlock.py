import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class CallBlock(Stmt):
    """Like a macro without a name but a call instead.  `call` is called with
    the unnamed macro as `caller` argument this node holds.
    """
    fields = ('call', 'args', 'defaults', 'body')
    call: 'Call'
    args: t.List['Name']
    defaults: t.List['Expr']
    body: t.List[Node]