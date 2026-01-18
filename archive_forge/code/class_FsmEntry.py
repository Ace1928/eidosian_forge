import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Sequence, Type, TypeVar, Union
@dataclass
class FsmEntry(Generic[T_FsmInputs, T_FsmContext]):
    condition: Callable[[T_FsmInputs], bool]
    target_state: Type[FsmState[T_FsmInputs, T_FsmContext]]
    action: Optional[Callable[[T_FsmInputs], None]] = None