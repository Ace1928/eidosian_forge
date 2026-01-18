import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _token_led_filter(self, left):
    condition = self._expression(0)
    self._match('rbracket')
    if self._current_token() == 'flatten':
        right = ast.identity()
    else:
        right = self._parse_projection_rhs(self.BINDING_POWER['filter'])
    return ast.filter_projection(left, right, condition)