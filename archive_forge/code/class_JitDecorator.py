from abc import ABC, abstractmethod
from numba.core.registry import DelayedRegistry, CPUDispatcher
from numba.core.decorators import jit
from numba.core.errors import InternalTargetMismatchError, NumbaValueError
from threading import local as tls
class JitDecorator(ABC):

    @abstractmethod
    def __call__(self):
        return NotImplemented