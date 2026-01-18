import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Sequence, Type, TypeVar, Union
@runtime_checkable
class FsmStateEnterWithContext(Protocol[T_FsmInputs, T_FsmContext_contra]):

    @abstractmethod
    def on_enter(self, inputs: T_FsmInputs, context: T_FsmContext_contra) -> None:
        ...