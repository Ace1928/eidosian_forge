from __future__ import annotations
import json
from typing import TYPE_CHECKING
import pyarrow
from pandas.compat import pa_version_under14p1
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays.interval import VALID_CLOSED
class ArrowIntervalType(pyarrow.ExtensionType):

    def __init__(self, subtype, closed: IntervalClosedType) -> None:
        assert closed in VALID_CLOSED
        self._closed: IntervalClosedType = closed
        if not isinstance(subtype, pyarrow.DataType):
            subtype = pyarrow.type_for_alias(str(subtype))
        self._subtype = subtype
        storage_type = pyarrow.struct([('left', subtype), ('right', subtype)])
        pyarrow.ExtensionType.__init__(self, storage_type, 'pandas.interval')

    @property
    def subtype(self):
        return self._subtype

    @property
    def closed(self) -> IntervalClosedType:
        return self._closed

    def __arrow_ext_serialize__(self) -> bytes:
        metadata = {'subtype': str(self.subtype), 'closed': self.closed}
        return json.dumps(metadata).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized) -> ArrowIntervalType:
        metadata = json.loads(serialized.decode())
        subtype = pyarrow.type_for_alias(metadata['subtype'])
        closed = metadata['closed']
        return ArrowIntervalType(subtype, closed)

    def __eq__(self, other):
        if isinstance(other, pyarrow.BaseExtensionType):
            return type(self) == type(other) and self.subtype == other.subtype and (self.closed == other.closed)
        else:
            return NotImplemented

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((str(self), str(self.subtype), self.closed))

    def to_pandas_dtype(self) -> IntervalDtype:
        return IntervalDtype(self.subtype.to_pandas_dtype(), self.closed)