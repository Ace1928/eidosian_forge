import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _token_nud_not(self, token):
    expr = self._expression(self.BINDING_POWER['not'])
    return ast.not_expression(expr)