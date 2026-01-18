import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _parse_comparator(self, left, comparator):
    right = self._expression(self.BINDING_POWER[comparator])
    return ast.comparator(comparator, left, right)