import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': ['array-number']})
def _func_avg(self, arg):
    if arg:
        return sum(arg) / float(len(arg))
    else:
        return None