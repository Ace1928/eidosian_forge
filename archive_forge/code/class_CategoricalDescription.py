from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, TypedDict
from .utils import ColumnNullType, DlpackDeviceType, DTypeKind
class CategoricalDescription(TypedDict):
    is_ordered: bool
    is_dictionary: bool
    categories: Optional['ProtocolColumn']