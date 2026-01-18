import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': ['array-number', 'array-string']})
def _func_min(self, arg):
    if arg:
        return min(arg)
    else:
        return None