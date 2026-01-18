import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _parse_multi_select_hash(self):
    pairs = []
    while True:
        key_token = self._lookahead_token(0)
        self._match_multiple_tokens(token_types=['quoted_identifier', 'unquoted_identifier'])
        key_name = key_token['value']
        self._match('colon')
        value = self._expression(0)
        node = ast.key_val_pair(key_name=key_name, node=value)
        pairs.append(node)
        if self._current_token() == 'comma':
            self._match('comma')
        elif self._current_token() == 'rbrace':
            self._match('rbrace')
            break
    return ast.multi_select_dict(nodes=pairs)