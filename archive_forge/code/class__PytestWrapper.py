from __future__ import annotations
import dataclasses
import enum
import functools
import inspect
from inspect import Parameter
from inspect import signature
import os
from pathlib import Path
import sys
from typing import Any
from typing import Callable
from typing import Final
from typing import NoReturn
import py
@dataclasses.dataclass
class _PytestWrapper:
    """Dummy wrapper around a function object for internal use only.

    Used to correctly unwrap the underlying function object when we are
    creating fixtures, because we wrap the function object ourselves with a
    decorator to issue warnings when the fixture function is called directly.
    """
    obj: Any