import dis
import enum
import opcode as _opcode
import sys
from abc import abstractmethod
from dataclasses import dataclass
from marshal import dumps as _dumps
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union
import bytecode as _bytecode
class TryEnd:
    __slots__ = 'entry'

    def __init__(self, entry: TryBegin) -> None:
        self.entry: TryBegin = entry

    def copy(self) -> 'TryEnd':
        return TryEnd(self.entry)