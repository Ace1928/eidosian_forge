import typing as t
from enum import Enum, auto
class TrieResult(Enum):
    FAILED = auto()
    PREFIX = auto()
    EXISTS = auto()