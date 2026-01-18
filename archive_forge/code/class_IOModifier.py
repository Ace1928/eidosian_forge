import enum
from typing import Optional, List, Union, Iterable, Tuple
class IOModifier(enum.Enum):
    """IO Modifier object"""
    INPUT = enum.auto()
    OUTPUT = enum.auto()