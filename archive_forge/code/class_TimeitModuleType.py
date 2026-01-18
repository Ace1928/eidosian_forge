from typing import Any, Callable, Dict, Protocol, runtime_checkable
@runtime_checkable
class TimeitModuleType(Protocol):
    """Modules generated from `timeit_template.cpp`."""

    def timeit(self, number: int) -> float:
        ...