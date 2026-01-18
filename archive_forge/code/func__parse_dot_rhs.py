import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _parse_dot_rhs(self, binding_power):
    lookahead = self._current_token()
    if lookahead in ['quoted_identifier', 'unquoted_identifier', 'star']:
        return self._expression(binding_power)
    elif lookahead == 'lbracket':
        self._match('lbracket')
        return self._parse_multi_select_list()
    elif lookahead == 'lbrace':
        self._match('lbrace')
        return self._parse_multi_select_hash()
    else:
        t = self._lookahead_token(0)
        allowed = ['quoted_identifier', 'unquoted_identifier', 'lbracket', 'lbrace']
        msg = 'Expecting: %s, got: %s' % (allowed, t['type'])
        self._raise_parse_error_for_token(t, msg)