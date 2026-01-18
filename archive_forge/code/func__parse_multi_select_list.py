import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _parse_multi_select_list(self):
    expressions = []
    while True:
        expression = self._expression()
        expressions.append(expression)
        if self._current_token() == 'rbracket':
            break
        else:
            self._match('comma')
    self._match('rbracket')
    return ast.multi_select_list(expressions)