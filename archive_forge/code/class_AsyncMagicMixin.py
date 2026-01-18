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
class AsyncMagicMixin(MagicMixin):

    def __init__(self, /, *args, **kw):
        self._mock_set_magics()
        _safe_super(AsyncMagicMixin, self).__init__(*args, **kw)
        self._mock_set_magics()