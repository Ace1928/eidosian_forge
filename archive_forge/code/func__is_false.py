import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def _is_false(self, value):
    return value == '' or value == [] or value == {} or (value is None) or (value is False)