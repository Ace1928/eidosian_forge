from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Union
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode
class CursorShape(Enum):
    _NEVER_CHANGE = '_NEVER_CHANGE'
    BLOCK = 'BLOCK'
    BEAM = 'BEAM'
    UNDERLINE = 'UNDERLINE'
    BLINKING_BLOCK = 'BLINKING_BLOCK'
    BLINKING_BEAM = 'BLINKING_BEAM'
    BLINKING_UNDERLINE = 'BLINKING_UNDERLINE'