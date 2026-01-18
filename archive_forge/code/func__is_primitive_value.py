import functools
import operator
def _is_primitive_value(obj):
    if obj is None:
        return False
    if isinstance(obj, (list, dict)):
        raise ValueError('query params may not contain repeated dicts or lists')
    return True