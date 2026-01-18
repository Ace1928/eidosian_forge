import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _expression(self, binding_power=0):
    left_token = self._lookahead_token(0)
    self._advance()
    nud_function = getattr(self, '_token_nud_%s' % left_token['type'], self._error_nud_token)
    left = nud_function(left_token)
    current_token = self._current_token()
    while binding_power < self.BINDING_POWER[current_token]:
        led = getattr(self, '_token_led_%s' % current_token, None)
        if led is None:
            error_token = self._lookahead_token(0)
            self._error_led_token(error_token)
        else:
            self._advance()
            left = led(left)
            current_token = self._current_token()
    return left