import dataclasses
from dataclasses import field
from types import CodeType, ModuleType
from typing import Any, Dict
@classmethod
def _is_excl(cls, mod):
    return any((mod.__name__ == excl for excl in cls.MOD_EXCLUDES))