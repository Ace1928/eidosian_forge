from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
class Remediation(NamedTuple):
    name: str
    immediate_msg: Optional[str] = None
    necessary_msg: Optional[str] = None
    necessary_fn: Optional[Callable[[Any], Any]] = None
    optional_msg: Optional[str] = None
    optional_fn: Optional[Callable[[Any], Any]] = None
    error_msg: Optional[str] = None