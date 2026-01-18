from __future__ import annotations
import os
import re
import sys
from collections import namedtuple
from typing import Any, cast
from types import ModuleType  # noqa
class module(ModuleType):
    """Customized Python module."""

    def __getattr__(self, name: str) -> Any:
        if name in object_origins:
            module = __import__(object_origins[name], None, None, [name])
            for extra_name in all_by_module[module.__name__]:
                setattr(self, extra_name, getattr(module, extra_name))
            return getattr(module, name)
        return ModuleType.__getattribute__(self, name)

    def __dir__(self) -> list[str]:
        result = list(new_module.__all__)
        result.extend(('__file__', '__path__', '__doc__', '__all__', '__docformat__', '__name__', '__path__', 'VERSION', '__package__', '__version__', '__author__', '__contact__', '__homepage__', '__docformat__'))
        return result