import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def _handle_ApplicationExpression(self, expression, command, x, y):
    function, args = expression.uncurry()
    if not isinstance(function, DrtAbstractVariableExpression):
        function = expression.function
        args = [expression.argument]
    function_bottom = self._visit(function, x, y)[1]
    max_bottom = max([function_bottom] + [self._visit(arg, x, y)[1] for arg in args])
    line_height = max_bottom - y
    function_drawing_top = self._get_centered_top(y, line_height, function._drawing_height)
    right = self._handle(function, command, x, function_drawing_top)[0]
    centred_string_top = self._get_centered_top(y, line_height, self._get_text_height())
    right = command(DrtTokens.OPEN, right, centred_string_top)[0]
    for i, arg in enumerate(args):
        arg_drawing_top = self._get_centered_top(y, line_height, arg._drawing_height)
        right = self._handle(arg, command, right, arg_drawing_top)[0]
        if i + 1 < len(args):
            right = command(DrtTokens.COMMA + ' ', right, centred_string_top)[0]
    right = command(DrtTokens.CLOSE, right, centred_string_top)[0]
    return (right, max_bottom)