import dataclasses
import inspect
import sys
from typing import Any, Callable, Tuple
import torch
class RegistrationHandle:
    """Does something when someone calls .destroy() on it"""

    def __init__(self, on_destroy: Callable):
        self._on_destroy = on_destroy

    def destroy(self) -> None:
        self._on_destroy()