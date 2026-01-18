import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def _handle_BinaryExpression(self, expression, command, x, y):
    first_height = self._visit(expression.first, 0, 0)[1]
    second_height = self._visit(expression.second, 0, 0)[1]
    line_height = max(first_height, second_height)
    centred_string_top = self._get_centered_top(y, line_height, self._get_text_height())
    right = command(DrtTokens.OPEN, x, centred_string_top)[0]
    first_height = expression.first._drawing_height
    right, first_bottom = self._handle(expression.first, command, right, self._get_centered_top(y, line_height, first_height))
    right = command(' %s ' % expression.getOp(), right, centred_string_top)[0]
    second_height = expression.second._drawing_height
    right, second_bottom = self._handle(expression.second, command, right, self._get_centered_top(y, line_height, second_height))
    right = command(DrtTokens.CLOSE, right, centred_string_top)[0]
    return (right, max(first_bottom, second_bottom))