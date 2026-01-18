from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, TypedDict
from .utils import ColumnNullType, DlpackDeviceType, DTypeKind
class ColumnBuffers(TypedDict):
    data: Tuple['ProtocolBuffer', Any]
    validity: Optional[Tuple['ProtocolBuffer', Any]]
    offsets: Optional[Tuple['ProtocolBuffer', Any]]