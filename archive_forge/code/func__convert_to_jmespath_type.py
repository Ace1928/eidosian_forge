import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
def _convert_to_jmespath_type(self, pyobject):
    return TYPES_MAP.get(pyobject, 'unknown')