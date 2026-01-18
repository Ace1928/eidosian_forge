import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _parse_index_expression(self):
    if self._lookahead(0) == 'colon' or self._lookahead(1) == 'colon':
        return self._parse_slice_expression()
    else:
        node = ast.index(self._lookahead_token(0)['value'])
        self._advance()
        self._match('rbracket')
        return node