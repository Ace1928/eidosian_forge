import typing
from typing import Callable, Optional
from mypy.plugin import FunctionContext, Plugin  # pylint: disable=no-name-in-module
from mypy.types import CallableType, Type, get_proper_type  # pylint: disable=no-name-in-module
class CrochetMypyPlugin(Plugin):
    """
    Assists mypy with type checking APIs not (yet) fully covered by Python's
    type hint annotation types, by copying run_in_reactor decorated function's
    argument types to the type mypy deduces for the wrapped function.
    """

    def get_function_hook(self, fullname: str) -> Optional[Callable[[FunctionContext], Type]]:
        if fullname == 'crochet.run_in_reactor':
            return _copyargs_callback
        return None