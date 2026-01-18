from typing import Any
import sys
from typing import _type_check  # type: ignore
def DefaultNamedArg(type=Any, name=None):
    """A keyword-only argument with a default value"""
    return type