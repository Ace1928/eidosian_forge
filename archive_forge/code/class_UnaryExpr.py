import inspect
import operator
import typing as t
from collections import deque
from markupsafe import Markup
from .utils import _PassArg
class UnaryExpr(Expr):
    """Baseclass for all unary expressions."""
    fields = ('node',)
    node: Expr
    operator: str
    abstract = True

    def as_const(self, eval_ctx: t.Optional[EvalContext]=None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)
        if eval_ctx.environment.sandboxed and self.operator in eval_ctx.environment.intercepted_unops:
            raise Impossible()
        f = _uaop_to_func[self.operator]
        try:
            return f(self.node.as_const(eval_ctx))
        except Exception as e:
            raise Impossible() from e