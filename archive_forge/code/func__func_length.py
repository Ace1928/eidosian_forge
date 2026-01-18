import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': ['string', 'array', 'object']})
def _func_length(self, arg):
    return len(arg)