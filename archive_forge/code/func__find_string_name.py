from jedi import debug
from jedi.inference.base_value import ValueSet, NO_VALUES, ValueWrapper
from jedi.inference.gradual.base import BaseTypingValue
def _find_string_name(self, lazy_value):
    if lazy_value is None:
        return None
    value_set = lazy_value.infer()
    if not value_set:
        return None
    if len(value_set) > 1:
        debug.warning('Found multiple values for a type variable: %s', value_set)
    name_value = next(iter(value_set))
    try:
        method = name_value.get_safe_value
    except AttributeError:
        return None
    else:
        safe_value = method(default=None)
        if isinstance(safe_value, str):
            return safe_value
        return None