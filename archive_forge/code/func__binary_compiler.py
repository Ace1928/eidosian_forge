from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def _binary_compiler(tmpl):
    """Compiler factory for the `_Compiler`."""
    return lambda self, left, right: tmpl % (self.compile(left), self.compile(right))