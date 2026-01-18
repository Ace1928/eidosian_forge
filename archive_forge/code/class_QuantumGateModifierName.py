import enum
from typing import Optional, List, Union, Iterable, Tuple
class QuantumGateModifierName(enum.Enum):
    """The names of the allowed modifiers of quantum gates."""
    CTRL = enum.auto()
    NEGCTRL = enum.auto()
    INV = enum.auto()
    POW = enum.auto()