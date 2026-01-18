from typing import Any, Callable, Dict, Protocol, runtime_checkable
class CallgrindModuleType(Protocol):
    """Replicates the valgrind endpoints in `torch._C`.

    These bindings are used to collect Callgrind profiles on earlier versions
    of PyTorch and will eventually be removed.
    """
    __file__: str
    __name__: str

    def _valgrind_supported_platform(self) -> bool:
        ...

    def _valgrind_toggle(self) -> None:
        ...