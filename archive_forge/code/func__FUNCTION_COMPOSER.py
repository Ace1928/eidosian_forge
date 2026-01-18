import re
from typing import Any, Callable, Dict, Tuple
from langchain.chains.query_constructor.ir import (
def _FUNCTION_COMPOSER(op_name: str) -> Callable:
    """
    Composer for functions.

    Args:
        op_name: Name of the function.

    Returns:
        Callable that takes a list of arguments and returns a string.
    """

    def f(*args: Any) -> str:
        args_: map[str] = map(str, args)
        return f'{op_name}({','.join(args_)})'
    return f