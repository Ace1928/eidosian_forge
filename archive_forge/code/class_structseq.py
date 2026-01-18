from __future__ import annotations
import types
from collections.abc import Hashable
from typing import (
from typing_extensions import NamedTuple  # Generic NamedTuple: Python 3.11+
from typing_extensions import OrderedDict  # Generic OrderedDict: Python 3.7.2+
from typing_extensions import Self  # Python 3.11+
from typing_extensions import TypeAlias  # Python 3.10+
from typing_extensions import Final, Protocol, runtime_checkable  # Python 3.8+
from optree import _C
from optree._C import PyTreeKind, PyTreeSpec
from optree._C import (
class structseq(tuple, Generic[_T_co], metaclass=_StructSequenceMeta):
    """A generic type stub for CPython's ``PyStructSequence`` type."""
    n_fields: Final[int]
    n_sequence_fields: Final[int]
    n_unnamed_fields: Final[int]

    def __init_subclass__(cls) -> NoReturn:
        """Prohibit subclassing."""
        raise TypeError("type 'structseq' is not an acceptable base type")

    def __new__(cls, sequence: Iterable[_T_co], dict: dict[str, Any]=...) -> Self:
        raise NotImplementedError