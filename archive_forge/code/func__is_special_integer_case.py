import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def _is_special_integer_case(x, y):
    if type(x) is int and (x == 0 or x == 1):
        return y is True or y is False
    elif type(y) is int and (y == 0 or y == 1):
        return x is True or x is False