import functools
from inspect import signature
from .common_op_utils import _basic_validation
def _decorator_func(wrapped_func, op, op_table):
    """
    Decorator function to register the given ``op`` in the provided
    ``op_table``
    """

    @functools.wraps(wrapped_func)
    def wrapper(types, args, kwargs, process_group):
        _basic_validation(op, args, kwargs)
        return wrapped_func(types, args, kwargs, process_group)
    _register_op(op, wrapper, op_table)
    return wrapper