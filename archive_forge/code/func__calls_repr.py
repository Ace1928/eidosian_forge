import asyncio
import contextlib
import io
import inspect
import pprint
import sys
import builtins
import pkgutil
from asyncio import iscoroutinefunction
from types import CodeType, ModuleType, MethodType
from unittest.util import safe_repr
from functools import wraps, partial
from threading import RLock
def _calls_repr(self, prefix='Calls'):
    """Renders self.mock_calls as a string.

        Example: "
Calls: [call(1), call(2)]."

        If self.mock_calls is empty, an empty string is returned. The
        output will be truncated if very long.
        """
    if not self.mock_calls:
        return ''
    return f'\n{prefix}: {safe_repr(self.mock_calls)}.'