from typing import Any, Callable, Dict, Protocol, runtime_checkable
class TimerClass(Protocol):
    """This is the portion of the `timeit.Timer` API used by benchmark utils."""

    def __init__(self, stmt: str, setup: str, timer: Callable[[], float], globals: Dict[str, Any], **kwargs: Any) -> None:
        ...

    def timeit(self, number: int) -> float:
        ...