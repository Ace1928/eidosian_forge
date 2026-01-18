from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Union
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode
class CursorShapeConfig(ABC):

    @abstractmethod
    def get_cursor_shape(self, application: Application[Any]) -> CursorShape:
        """
        Return the cursor shape to be used in the current state.
        """