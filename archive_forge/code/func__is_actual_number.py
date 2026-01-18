import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def _is_actual_number(x):
    if x is True or x is False:
        return False
    return isinstance(x, Number)