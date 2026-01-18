import ast
import difflib
from collections.abc import Mapping
from typing import Any, Callable, Dict
class InterpretorError(ValueError):
    """
    An error raised when the interpretor cannot evaluate a Python expression, due to syntax error or unsupported
    operations.
    """
    pass