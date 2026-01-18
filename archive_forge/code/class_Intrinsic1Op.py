import dis
import enum
import opcode as _opcode
import sys
from abc import abstractmethod
from dataclasses import dataclass
from marshal import dumps as _dumps
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union
import bytecode as _bytecode
@enum.unique
class Intrinsic1Op(enum.IntEnum):
    INTRINSIC_1_INVALID = 0
    INTRINSIC_PRINT = 1
    INTRINSIC_IMPORT_STAR = 2
    INTRINSIC_STOPITERATION_ERROR = 3
    INTRINSIC_ASYNC_GEN_WRAP = 4
    INTRINSIC_UNARY_POSITIVE = 5
    INTRINSIC_LIST_TO_TUPLE = 6
    INTRINSIC_TYPEVAR = 7
    INTRINSIC_PARAMSPEC = 8
    INTRINSIC_TYPEVARTUPLE = 9
    INTRINSIC_SUBSCRIPT_GENERIC = 10
    INTRINSIC_TYPEALIAS = 11