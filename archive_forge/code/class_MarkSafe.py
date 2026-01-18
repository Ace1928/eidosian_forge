import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class MarkSafe(Expr):
    """Mark the wrapped expression as safe (wrap it as `Markup`)."""
    fields = ('expr',)
    expr: Expr

    def as_const(self, eval_ctx: t.Optional[EvalContext]=None) -> Markup:
        eval_ctx = get_eval_context(self, eval_ctx)
        return Markup(self.expr.as_const(eval_ctx))