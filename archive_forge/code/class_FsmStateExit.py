import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Sequence, Type, TypeVar, Union
@runtime_checkable
class FsmStateExit(Protocol[T_FsmInputs, T_FsmContext_cov]):

    @abstractmethod
    def on_exit(self, inputs: T_FsmInputs) -> T_FsmContext_cov:
        ...