import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _token_led_and(self, left):
    right = self._expression(self.BINDING_POWER['and'])
    return ast.and_expression(left, right)