import warnings
import weakref
from collections import UserDict
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import srsly
from ..errors import Errors, Warnings
from .span_group import SpanGroup
class SpanGroups(UserDict):
    """A dict-like proxy held by the Doc, to control access to span groups."""
    _EMPTY_BYTES = srsly.msgpack_dumps([])

    def __init__(self, doc: 'Doc', items: Iterable[Tuple[str, SpanGroup]]=tuple()) -> None:
        self.doc_ref = weakref.ref(doc)
        UserDict.__init__(self, items)

    def __setitem__(self, key: str, value: Union[SpanGroup, Iterable['Span']]) -> None:
        if not isinstance(value, SpanGroup):
            value = self._make_span_group(key, value)
        assert value.doc is self.doc_ref()
        UserDict.__setitem__(self, key, value)

    def _make_span_group(self, name: str, spans: Iterable['Span']) -> SpanGroup:
        doc = self._ensure_doc()
        return SpanGroup(doc, name=name, spans=spans)

    def copy(self, doc: Optional['Doc']=None) -> 'SpanGroups':
        if doc is None:
            doc = self._ensure_doc()
        data_copy = ((k, v.copy(doc=doc)) for k, v in self.items())
        return SpanGroups(doc, items=data_copy)

    def setdefault(self, key, default=None):
        if not isinstance(default, SpanGroup):
            if default is None:
                spans = []
            else:
                spans = default
            default = self._make_span_group(key, spans)
        return super().setdefault(key, default=default)

    def to_bytes(self) -> bytes:
        if len(self) == 0:
            return self._EMPTY_BYTES
        msg: Dict[bytes, List[str]] = {}
        for key, value in self.items():
            msg.setdefault(value.to_bytes(), []).append(key)
        return srsly.msgpack_dumps(msg)

    def from_bytes(self, bytes_data: bytes) -> 'SpanGroups':
        msg = [] if not bytes_data or bytes_data == self._EMPTY_BYTES else srsly.msgpack_loads(bytes_data)
        self.clear()
        doc = self._ensure_doc()
        if isinstance(msg, list):
            for value_bytes in msg:
                group = SpanGroup(doc).from_bytes(value_bytes)
                if group.name in self:
                    warnings.warn(Warnings.W120.format(group_name=group.name, group_values=self[group.name]))
                self[group.name] = group
        else:
            for value_bytes, keys in msg.items():
                group = SpanGroup(doc).from_bytes(value_bytes)
                self[keys[0]] = group
                for key in keys[1:]:
                    self[key] = group.copy()
        return self

    def _ensure_doc(self) -> 'Doc':
        doc = self.doc_ref()
        if doc is None:
            raise ValueError(Errors.E866)
        return doc