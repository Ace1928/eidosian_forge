import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Sequence, Type, TypeVar, Union
@runtime_checkable
class FsmStateStay(Protocol[T_FsmInputs]):

    @abstractmethod
    def on_stay(self, inputs: T_FsmInputs) -> None:
        ...