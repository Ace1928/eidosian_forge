import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': ['array-string', 'array-number']})
def _func_sort(self, arg):
    return list(sorted(arg))