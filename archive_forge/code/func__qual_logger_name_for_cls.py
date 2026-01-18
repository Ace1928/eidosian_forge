from __future__ import annotations
import logging
import sys
from typing import Any
from typing import Optional
from typing import overload
from typing import Set
from typing import Type
from typing import TypeVar
from typing import Union
from .util import py311
from .util import py38
from .util.typing import Literal
def _qual_logger_name_for_cls(cls: Type[Identified]) -> str:
    return getattr(cls, '_sqla_logger_namespace', None) or cls.__module__ + '.' + cls.__name__