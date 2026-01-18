import re
from typing import Any, Callable, Dict, Tuple
from langchain.chains.query_constructor.ir import (
def _DEFAULT_COMPOSER(op_name: str) -> Callable:
    """
    Default composer for logical operators.

    Args:
        op_name: Name of the operator.

    Returns:
        Callable that takes a list of arguments and returns a string.
    """

    def f(*args: Any) -> str:
        args_: map[str] = map(str, args)
        return f' {op_name} '.join(args_)
    return f