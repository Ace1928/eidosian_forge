import logging
import types
from typing import Any, Callable, Dict, Sequence, TypeVar
from .._abc import Instrument
def _public(fn: F) -> F:
    return fn