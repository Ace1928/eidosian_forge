import typing
from typing import Callable, Optional
from mypy.plugin import FunctionContext, Plugin  # pylint: disable=no-name-in-module
from mypy.types import CallableType, Type, get_proper_type  # pylint: disable=no-name-in-module
def _copyargs_callback(ctx: FunctionContext) -> Type:
    """
    Copy the parameters from the signature of the type of the argument of the
    call to the signature of the return type.
    """
    original_return_type = ctx.default_return_type
    if not ctx.arg_types or len(ctx.arg_types[0]) != 1:
        return original_return_type
    arg_type = get_proper_type(ctx.arg_types[0][0])
    default_return_type = get_proper_type(original_return_type)
    if not (isinstance(arg_type, CallableType) and isinstance(default_return_type, CallableType)):
        return original_return_type
    return default_return_type.copy_modified(arg_types=arg_type.arg_types, arg_kinds=arg_type.arg_kinds, arg_names=arg_type.arg_names, variables=arg_type.variables, is_ellipsis_args=arg_type.is_ellipsis_args)