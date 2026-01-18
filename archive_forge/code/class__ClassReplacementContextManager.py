from typing import Any, Mapping, Type, Union
import torch
from torch import Tensor
class _ClassReplacementContextManager:
    """A context manager to monkeypatch classes."""

    def __init__(self, mapping: Mapping[str, Type]) -> None:
        self._mapping = mapping
        self._originals = {}
        self._modules = {}
        for class_string in mapping:
            module_name, class_name = class_string.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            self._modules[class_string] = module
            self._originals[class_string] = getattr(module, class_name)

    def __enter__(self) -> None:
        for class_string, replacement in self._mapping.items():
            _, class_name = class_string.rsplit('.', 1)
            setattr(self._modules[class_string], class_name, replacement)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for class_string, replacement in self._mapping.items():
            _, class_name = class_string.rsplit('.', 1)
            setattr(self._modules[class_string], class_name, self._originals[class_string])