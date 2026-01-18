import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
@signature({'types': ['array']}, {'types': ['expref']})
def _func_sort_by(self, array, expref):
    if not array:
        return array
    required_type = self._convert_to_jmespath_type(type(expref.visit(expref.expression, array[0])).__name__)
    if required_type not in ['number', 'string']:
        raise exceptions.JMESPathTypeError('sort_by', array[0], required_type, ['string', 'number'])
    keyfunc = self._create_key_func(expref, [required_type], 'sort_by')
    return list(sorted(array, key=keyfunc))