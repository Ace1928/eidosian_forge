import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _token_led_lparen(self, left):
    if left['type'] != 'field':
        prev_t = self._lookahead_token(-2)
        raise exceptions.ParseError(prev_t['start'], prev_t['value'], prev_t['type'], "Invalid function name '%s'" % prev_t['value'])
    name = left['value']
    args = []
    while not self._current_token() == 'rparen':
        expression = self._expression()
        if self._current_token() == 'comma':
            self._match('comma')
        args.append(expression)
    self._match('rparen')
    function_node = ast.function_expression(name, args)
    return function_node