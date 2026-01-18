import dataclasses
import sys
import types
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Protocol, Union
from typing_extensions import TypeAlias
import torch
class DynamoGuardHook(Protocol):

    def __call__(self, guard_fn: GuardFn, code: types.CodeType, f_locals: Dict[str, object], index: int, last: bool) -> None:
        ...