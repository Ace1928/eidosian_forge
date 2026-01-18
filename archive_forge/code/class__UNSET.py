import dis
import enum
import opcode as _opcode
import sys
from abc import abstractmethod
from dataclasses import dataclass
from marshal import dumps as _dumps
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union
import bytecode as _bytecode
class _UNSET(int):
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __eq__(self, other) -> bool:
        return self is other