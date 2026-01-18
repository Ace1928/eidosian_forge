import functools
import inspect
import wrapt
from debtcollector import _utils
def _check_it(cls):
    if not inspect.isclass(cls):
        _qual, type_name = _utils.get_qualified_name(type(cls))
        raise TypeError("Unexpected class type '%s' (expected class type only)" % type_name)