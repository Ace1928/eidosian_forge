from __future__ import annotations
from collections.abc import Iterable
from dataclasses import make_dataclass
class DataBin(metaclass=DataBinMeta):
    """Base class for data bin containers.

    Subclasses are typically made via :class:`~make_data_bin`, which is a specialization of
    :class:`make_dataclass`.
    """
    _RESTRICTED_NAMES = ('_RESTRICTED_NAMES', '_SHAPE', '_FIELDS', '_FIELD_TYPES')
    _SHAPE: tuple[int, ...] | None = None
    _FIELDS: tuple[str, ...] = ()
    'The fields allowed in this data bin.'
    _FIELD_TYPES: tuple[type, ...] = ()
    'The types of each field.'

    def __len__(self):
        return len(self._FIELDS)

    def __repr__(self):
        vals = (f'{name}={getattr(self, name)}' for name in self._FIELDS if hasattr(self, name))
        return f'{type(self)}({', '.join(vals)})'