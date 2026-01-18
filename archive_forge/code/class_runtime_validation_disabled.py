import inspect
from functools import wraps
from typing import Any, Callable, Optional, Type, Union, get_type_hints
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from torch.utils.data.datapipes._typing import _DataPipeMeta
class runtime_validation_disabled:
    prev: bool

    def __init__(self) -> None:
        global _runtime_validation_enabled
        self.prev = _runtime_validation_enabled
        _runtime_validation_enabled = False

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _runtime_validation_enabled
        _runtime_validation_enabled = self.prev