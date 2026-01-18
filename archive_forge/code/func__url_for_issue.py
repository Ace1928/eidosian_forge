from __future__ import annotations
import sys
import warnings
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, TypeVar
import attrs
def _url_for_issue(issue: int) -> str:
    return f'https://github.com/python-trio/trio/issues/{issue}'