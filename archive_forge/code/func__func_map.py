import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': ['expref']}, {'types': ['array']})
def _func_map(self, expref, arg):
    result = []
    for element in arg:
        result.append(expref.visit(expref.expression, element))
    return result