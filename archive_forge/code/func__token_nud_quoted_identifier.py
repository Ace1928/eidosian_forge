import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _token_nud_quoted_identifier(self, token):
    field = ast.field(token['value'])
    if self._current_token() == 'lparen':
        t = self._lookahead_token(0)
        raise exceptions.ParseError(0, t['value'], t['type'], 'Quoted identifier not allowed for function names.')
    return field