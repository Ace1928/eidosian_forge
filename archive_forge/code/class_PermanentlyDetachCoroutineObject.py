from __future__ import annotations
import enum
import types
from typing import TYPE_CHECKING, Any, Callable, NoReturn
import attrs
import outcome
from . import _run
@attrs.frozen(slots=False)
class PermanentlyDetachCoroutineObject:
    final_outcome: outcome.Outcome[Any]